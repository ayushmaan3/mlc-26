"""Run E: fine-tuned e5 cosine as a pair FEATURE, leak-free (Fable review 2). Run C only used the fine-tuned model to add
candidates, because a model fine-tuned on train pairs scores those pairs better than unseen (test) pairs.
Fix: two models, split like the two XGBoost folds of s4_train (FOLDS = S1 buckets [0,2,4,6] / [1,3,5,7]):
  m{k} = original e5 fine-tuned (s17 recipe) on N_TRAIN (S1, match) pairs whose S1 bucket is in FOLDS[k].
  train pair in bucket j of fold f -> cosine from m{1-f}, a model that never saw bucket j.
  test pair -> both, as columns <feat>_m0 / <feat>_m1 (+ <feat> = m0). s4_train's stacked test path gives booster g
  (trained on FOLDS[g], whose rows carry m{1-g} cosines) the columns <feat>_m{1-g}: the same model it was trained with.
Features per union pair (s10_v3.union_pairs = keys + e5 kNN + fine-tuned kNN):
  ftc (cosine), ftc_rk / ftc_gap (rank / gap to the best within the S1), ftc_rk_id / ftc_gap_id (within the S2/S3 record's
  candidates; on train each pair keeps its own leak-free cosine, so a record's rivals may come from the other model).
Steps (each skips finished outputs):
  train <k>      : fine-tune m{k} -> WORK/e5ft_m{k}; recall check on held-out S1 of the OTHER fold (must beat the original e5)
  encode <split> : every record with m0 and m1, tokenized once (m0 on cuda:0, m1 on cuda:1 if present)
                   -> WORK/{split}_embft{k}.f16 (rows in length order) + {split}_embft_pos.npy (records row -> memmap row)
  feats <split>  : -> WORK/ftc_{split}_{j}.parquet
usage: s18_ftcos.py train <k> | encode <split> | feats <split>
env: BER_E5 (original e5 dir), BER_N_TRAIN (300k), BER_N_EVAL (20k), BER_N_DB (300k)"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from common import WORK, done, timer
from s2_block import BUCKETS
from s4_train import FOLDS
from s10_v3 import union_pairs
import s17_ft_retrieval as s17

E5 = os.environ.get("BER_E5", "/kaggle/input/multilingual-e5-small")
N_TRAIN = int(os.environ.get("BER_N_TRAIN", 300_000))
N_EVAL = int(os.environ.get("BER_N_EVAL", 20_000))
N_DB = int(os.environ.get("BER_N_DB", 300_000))
DIM, MAXLEN, BS = 384, 64, 1024
FEATS = ["ftc", "ftc_rk", "ftc_gap", "ftc_rk_id", "ftc_gap_id"]


def model_dir(k):
    return WORK / f"e5ft_m{k}"


def emb_path(split, k):
    return WORK / f"{split}_embft{k}.f16"


def pos_path(split):
    return WORK / f"{split}_embft_pos.npy"


def ftc_path(split, j):
    return WORK / f"ftc_{split}_{j}.parquet"


def fold_of(j):
    return 0 if j in FOLDS[0] else 1


TXT = ("query: " + pl.col("name") + " | " + pl.col("addr")).str.slice(0, 250)   # same text as e1_emb / s17
MIN_GAIN = float(os.environ.get("BER_FT_MIN_GAIN", 0.0))   # recall@20 gain a fold model must show (local smoke: -1)


def train(k):
    if done(model_dir(k) / "model.safetensors"):
        return
    with timer(f"fine-tune m{k} on S1 buckets {FOLDS[k]}"):
        rec = pl.read_parquet(WORK / "train_records.parquet", columns=["id", "src", "name", "addr", "country"])
        rec = rec.select("id", "src", pl.col("country").cast(pl.Utf8), TXT.alias("t"))
        gt = pl.read_parquet(WORK / "train_gt.parquet")
        mine = (pl.col("s1") % BUCKETS).is_in(FOLDS[k])
        tr = gt.filter(mine).sample(min(N_TRAIN, gt.filter(mine).height), seed=11 + k)
        s1_eval = gt.filter(~mine).select("s1").unique()
        s1_eval = s1_eval.sample(min(N_EVAL, s1_eval.height), seed=21 + k)
        truth = gt.join(s1_eval, on="s1", how="semi")
        q = rec.join(s1_eval.rename({"s1": "id"}), on="id", how="semi")
        others = rec.filter(pl.col("src") != 1).join(truth.select("id"), on="id", how="anti")
        db = pl.concat([rec.join(truth.select("id"), on="id", how="semi"),
                        others.sample(min(N_DB, others.height), seed=31 + k)]).select("id", "country", "t")
        s17.log(f"m{k}: train pairs {tr.height:,} | eval S1 {q.height:,} (other fold) | eval pairs {truth.height:,} | db {db.height:,}")
        pairs = (tr.join(rec.select(pl.col("id").alias("s1"), pl.col("t").alias("ta")), on="s1")
                 .join(rec.select("id", pl.col("t").alias("tb")), on="id"))
        a_txt, b_txt = pairs["ta"].to_list(), pairs["tb"].to_list()
        del rec, pairs
        tok = AutoTokenizer.from_pretrained(E5)
        model = AutoModel.from_pretrained(E5).to(s17.DEV)
        base = s17.recall_at(model, tok, q, db, truth)
        s17.log(f"m{k} BASELINE original e5 recall@5/10/20: {base}")
        s17.finetune(model, tok, a_txt, b_txt, seed=41 + k)
        ft = s17.recall_at(model, tok, q, db, truth)
        s17.log(f"m{k} FINE-TUNED recall@5/10/20: {ft}  (delta @20 {ft[20] - base[20]:+.4f})")
        assert ft[20] - base[20] > MIN_GAIN, f"fine-tuned m{k} is not better than the original e5 -> refusing to use it"
        tmp = WORK / f"e5ft_m{k}.part"
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        os.replace(tmp, model_dir(k))


@torch.no_grad()
def encode(split):
    if all(done(emb_path(split, k)) for k in (0, 1)) and done(pos_path(split)):
        return
    with timer(f"encode {split} with m0 + m1"):
        r = pl.read_parquet(WORK / f"{split}_records.parquet", columns=["id", "name", "addr"])
        txt = r.select(TXT).to_series().to_list()
        del r
        n = len(txt)
        order = np.argsort(np.fromiter((len(t) for t in txt), dtype=np.int32, count=n), kind="stable")
        ng = torch.cuda.device_count()
        devs = [f"cuda:{min(k, ng - 1)}" if ng else "cpu" for k in (0, 1)]
        tok = AutoTokenizer.from_pretrained(model_dir(0))
        models = [AutoModel.from_pretrained(model_dir(k)).to(devs[k]).half().eval() for k in (0, 1)]
        outs = [np.memmap(emb_path(split, k).with_suffix(".part"), dtype=np.float16, mode="w+", shape=(n, DIM)) for k in (0, 1)]
        t0 = time.time()
        for s in range(0, n, BS):
            idx = order[s:s + BS]
            enc = tok([txt[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            es = []
            for k in (0, 1):
                e = {a: b.to(devs[k], non_blocking=True) for a, b in enc.items()}
                h = models[k](**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
                es.append(torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=1).half())
            for k in (0, 1):
                outs[k][s:s + len(idx)] = es[k].cpu().numpy()   # memmap row = position in length order
            if s % (BS * 500) == 0:
                el = time.time() - t0
                print(f"  {s:,}/{n:,} {(s + len(idx)) / max(el, 1e-6):.0f}/s eta {el / (s + len(idx)) * (n - s) / 60:.0f} min", flush=True)
        for k in (0, 1):
            outs[k].flush()
        del outs
        pos = np.empty(n, dtype=np.int64)
        pos[order] = np.arange(n, dtype=np.int64)   # records row -> memmap row
        np.save(pos_path(split).with_suffix(".part.npy"), pos)
        for k in (0, 1):
            os.replace(emb_path(split, k).with_suffix(".part"), emb_path(split, k))
        os.replace(pos_path(split).with_suffix(".part.npy"), pos_path(split))


def pair_cos(E, u, dev):
    """cosine(S1, candidate) for pairs u [p1, p2] (memmap rows); u must be sorted by p2 (near-sequential reads)."""
    s1_rows = np.unique(u["p1"].to_numpy())
    S = torch.from_numpy(np.ascontiguousarray(E[s1_rows])).to(dev).float()
    loc = np.searchsorted(s1_rows, u["p1"].to_numpy()).astype(np.int64)
    p2 = u["p2"].to_numpy().astype(np.int64)
    out = np.empty(u.height, dtype=np.float32)
    for i in range(0, u.height, 500_000):
        b = torch.from_numpy(np.ascontiguousarray(E[p2[i:i + 500_000]])).to(dev).float()
        out[i:i + 500_000] = (S[torch.from_numpy(loc[i:i + 500_000]).to(dev)] * b).sum(1).cpu().numpy()
    del S
    return out


def derived(c, sfx):
    """rank / gap within the S1 and within the record, for cosine column c; output names FEATS + sfx."""
    return [pl.col(c).alias("ftc" + sfx),
            pl.col(c).rank("min", descending=True).over("s1").cast(pl.Int16).alias("ftc_rk" + sfx),
            (pl.col(c).max().over("s1") - pl.col(c)).cast(pl.Float32).alias("ftc_gap" + sfx),
            pl.col(c).rank("min", descending=True).over("id").cast(pl.Int16).alias("ftc_rk_id" + sfx),
            (pl.col(c).max().over("id") - pl.col(c)).cast(pl.Float32).alias("ftc_gap_id" + sfx)]


def feats(split):
    if all(done(ftc_path(split, j)) for j in range(BUCKETS)):
        return
    with timer(f"ftc features {split}"):
        n = len(np.load(pos_path(split)))
        Es = [np.memmap(emb_path(split, k), dtype=np.float16, mode="r", shape=(n, DIM)) for k in (0, 1)]
        ids = pl.read_parquet(WORK / f"{split}_records.parquet", columns=["id"])["id"]
        row = pl.DataFrame({"id": ids, "p": np.load(pos_path(split))})
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        parts = []
        for j in range(BUCKETS):
            u = (union_pairs(split, j).join(row.rename({"id": "s1", "p": "p1"}), on="s1")
                 .join(row.rename({"p": "p2"}), on="id").sort("p2"))
            # train: one leak-free cosine per pair (model of the other fold); test: both models
            ks = [1 - fold_of(j)] if split == "train" else [0, 1]
            u = u.with_columns([pl.Series(f"c{k}", pair_cos(Es[k], u, dev)) for k in ks])
            u = u.select("s1", "id", *[f"c{k}" for k in ks])
            if split == "train":
                u = u.rename({f"c{ks[0]}": "c"})
            parts.append(u)
            print(f"  cosines bucket {j}: {u.height:,} pairs", flush=True)
        x = pl.concat(parts)
        del parts
        if split == "train":
            x = x.select("s1", "id", *derived("c", ""))
        else:
            x = x.select("s1", "id", *derived("c0", "_m0"), *derived("c1", "_m1"))
            x = x.with_columns([pl.col(f + "_m0").alias(f) for f in FEATS])   # base column (renamed per booster in s4_train)
        for j in range(BUCKETS):
            tmp = ftc_path(split, j).with_suffix(".tmp")
            x.filter(pl.col("s1") % BUCKETS == j).write_parquet(tmp)
            os.replace(tmp, ftc_path(split, j))
        del x


if __name__ == "__main__":
    step, arg = sys.argv[1], sys.argv[2]
    {"train": lambda a: train(int(a)), "encode": encode, "feats": feats}[step](arg)
