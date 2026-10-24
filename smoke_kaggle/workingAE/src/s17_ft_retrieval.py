"""Quick validation: does fine-tuning e5-small on our own train pairs improve kNN retrieval (blocking recall)?
Plain PyTorch (no sentence-transformers). Text = "query: <name> | <address>" (same as the current kNN).
  1. held-out S1 = bucket 7 (s1 % 8 == 7), sample N_EVAL; training pairs = (S1, matched S2/S3) from the other buckets, sample N_TRAIN
  2. eval database = all true matches of the held-out S1 + N_DB random other S2/S3 records (same countries)
  3. baseline: original e5 -> within-country kNN@20 recall on the held-out true pairs
  4. fine-tune 1 epoch, in-batch negatives (MultipleNegativesRanking, symmetric, scale 20), fp16, max_len 64
  5. fine-tuned e5 -> same recall. Saves the fine-tuned model to OUT_MODEL.
env: BER_DATA_DIR (train_records/train_gt), BER_E5 (model dir), BER_FT_OUT, BER_N_TRAIN, BER_N_EVAL, BER_N_DB"""
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

DATA = os.environ.get("BER_DATA_DIR", "/content/work")
E5 = os.environ.get("BER_E5", "/content/drive/MyDrive/MLC'26/models/multilingual-e5-small")
OUT_MODEL = os.environ.get("BER_FT_OUT", "/content/e5_ft")
N_TRAIN = int(os.environ.get("BER_N_TRAIN", 300_000))
N_EVAL = int(os.environ.get("BER_N_EVAL", 100_000))
N_DB = int(os.environ.get("BER_N_DB", 800_000))
BS, LR, MAXLEN, SCALE, K = 128, 2e-5, 64, 20.0, 20
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@torch.no_grad()
def encode(model, tok, texts, bs=512):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16)
    for k in range(0, len(texts), bs):
        idx = order[k:k + bs]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=1)
        out[torch.from_numpy(idx)] = e.half().cpu()
    return out


def recall_at(model, tok, q, db, truth):
    """q/db: frames [id, country, t]; truth [s1, id]. Within-country kNN; returns recall@5/10/20."""
    Eq, Ed = encode(model, tok, q["t"].to_list()), encode(model, tok, db["t"].to_list())
    hits = {5: 0, 10: 0, 20: 0}
    tset = set(zip(truth["s1"].to_list(), truth["id"].to_list()))
    for c in q["country"].unique().to_list():
        qi = np.where((q["country"] == c).to_numpy())[0]
        di = np.where((db["country"] == c).to_numpy())[0]
        if len(qi) == 0 or len(di) == 0:
            continue
        D = Ed[di].to(DEV)
        qid, did = q["id"].to_numpy()[qi], db["id"].to_numpy()[di]
        for s in range(0, len(qi), 2048):
            Q = Eq[qi[s:s + 2048]].to(DEV)
            top = (Q @ D.T).topk(min(K, len(di)), dim=1).indices.cpu().numpy()
            for r, row in enumerate(top):
                s1 = qid[s + r]
                for k in (5, 10, 20):
                    hits[k] += sum((s1, did[x]) in tset for x in row[:k])
    return {k: round(v / len(tset), 4) for k, v in hits.items()}


def finetune(model, tok, a_txt, b_txt, seed=3):
    """1 epoch, in-batch negatives (symmetric MultipleNegativesRanking, scale SCALE), fp16, linear warmup/decay. In place."""
    perm = np.random.default_rng(seed).permutation(len(a_txt))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(perm) // BS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps))) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    ce = torch.nn.CrossEntropyLoss()

    def emb(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)

    model.train()
    t0 = time.time()
    for s in range(steps):
        idx = perm[s * BS:(s + 1) * BS]
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            ea, eb = emb([a_txt[i] for i in idx]), emb([b_txt[i] for i in idx])
            sim = (ea.float() @ eb.float().T) * SCALE
        lab = torch.arange(len(idx), device=DEV)
        loss = (ce(sim, lab) + ce(sim.T, lab)) / 2   # duplicates of the same S1 in a batch are rare at this sample size
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if s % 200 == 0:
            el = time.time() - t0
            log(f"step {s}/{steps} loss {loss.item():.4f} eta {el / (s + 1) * (steps - s) / 60:.0f} min")
    model.eval()


def main():
    rec = pl.read_parquet(f"{DATA}/train_records.parquet", columns=["id", "src", "name", "addr", "country"])
    rec = rec.select("id", "src", pl.col("country").cast(pl.Utf8),
                     ("query: " + pl.col("name") + " | " + pl.col("addr")).str.slice(0, 250).alias("t"))
    gt = pl.read_parquet(f"{DATA}/train_gt.parquet")
    s1_eval = gt.filter(pl.col("s1") % 8 == 7).select("s1").unique()
    s1_eval = s1_eval.sample(min(N_EVAL, s1_eval.height), seed=0)
    truth = gt.join(s1_eval, on="s1", how="semi")
    tr = gt.filter(pl.col("s1") % 8 != 7).sample(min(N_TRAIN, gt.height), seed=1)
    q = rec.join(s1_eval.rename({"s1": "id"}), on="id", how="semi")
    others = rec.filter(pl.col("src") != 1).join(truth.select("id"), on="id", how="anti")
    db = pl.concat([rec.join(truth.select("id"), on="id", how="semi"),
                    others.sample(min(N_DB, others.height), seed=2)]).select("id", "country", "t")
    log(f"eval S1 {q.height:,} | eval true pairs {truth.height:,} | db {db.height:,} | train pairs {tr.height:,}")

    tok = AutoTokenizer.from_pretrained(E5)
    model = AutoModel.from_pretrained(E5).to(DEV)
    base = recall_at(model, tok, q, db, truth)
    log("BASELINE e5 recall@5/10/20:", base)

    pairs = (tr.join(rec.select(pl.col("id").alias("s1"), pl.col("t").alias("ta")), on="s1")
             .join(rec.select("id", pl.col("t").alias("tb")), on="id"))   # order irrelevant: shuffled below
    a_txt, b_txt = pairs["ta"].to_list(), pairs["tb"].to_list()
    del rec, pairs
    finetune(model, tok, a_txt, b_txt)
    model.save_pretrained(OUT_MODEL)
    tok.save_pretrained(OUT_MODEL)
    ft = recall_at(model, tok, q, db, truth)
    log("FINE-TUNED e5 recall@5/10/20:", ft)
    log(f"DELTA recall@20: {ft[20] - base[20]:+.4f}   (go if the fine-tuned model clearly beats the baseline)")


if __name__ == "__main__":
    main()
