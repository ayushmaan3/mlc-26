"""Stage 8: cross-encoder re-ranker on uncertain pairs (stage-1 OOF p in [LO, HI]).
Fine-tunes cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0) once per fold, on the same S1-bucket folds as
XGBoost, so train scores are out-of-fold. Test scores = mean of the two fold models.
Input text per record: raw "name | address" (native scripts kept; the model is multilingual).
Out: ce_{split}_{j}.parquet [s1, id, ce] (logit, Float32). Pairs outside [LO, HI] are not scored (NaN in stage 3).
Resumable: fold models and per-bucket outputs are skipped when present."""
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import WORK, done, timer
from s2_block import BUCKETS
from s4_train import FOLDS, ce_path, p_path

MODEL = os.environ.get("BER_CE_MODEL", "/content/drive/MyDrive/MLC'26/models/mmarco-mMiniLMv2-L12-H384-v1")
LO, HI = float(os.environ.get("BER_CE_LO", 0.03)), float(os.environ.get("BER_CE_HI", 0.97))
MAX_TRAIN = int(os.environ.get("BER_CE_TRAIN", 1_500_000))   # training pairs per fold
MAXLEN = int(os.environ.get("BER_CE_MAXLEN", 96))
BS, BS_INF, LR = 64, 512, 3e-5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fold_dir(f):
    return WORK / f"ce_model_f{f}"


def texts(split):
    return (pl.scan_parquet(WORK / f"{split}_records.parquet")
            .select("id", (pl.col("name") + " | " + pl.col("addr")).str.slice(0, 300).alias("t")))


def uncertain(split, j):
    return pl.scan_parquet(p_path(1, split, j)).filter(pl.col("p").is_between(LO, HI)).select("s1", "id")


def with_text(pairs, split):
    t = texts(split)
    return (pairs.join(t.rename({"id": "s1", "t": "a"}), on="s1").join(t.rename({"t": "b"}), on="id")
            .collect())


def train_fold(f):
    if done(fold_dir(f) / "config.json"):
        return
    with timer(f"ce train fold {f}"):
        gt = pl.scan_parquet(WORK / "train_gt.parquet").with_columns(pl.lit(1.0, pl.Float32).alias("y"))
        pairs = (pl.concat([uncertain("train", j) for j in FOLDS[f]]).join(gt, on=["s1", "id"], how="left")
                 .with_columns(pl.col("y").fill_null(0.0)))
        n = pairs.select(pl.len()).collect().item()
        pairs = with_text(pairs, "train").sample(min(n, MAX_TRAIN), seed=f, shuffle=True)
        print(f"fold {f}: {n:,} uncertain pairs, training on {pairs.height:,}, pos rate {pairs['y'].mean():.3f}",
              flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(DEV)
        steps = math.ceil(pairs.height / BS)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
        lossf = torch.nn.BCEWithLogitsLoss()
        a, b, y = pairs["a"].to_list(), pairs["b"].to_list(), pairs["y"].to_numpy()
        del pairs
        model.train()
        t0, run = time.time(), 0.0
        for k in range(steps):
            sl = slice(k * BS, (k + 1) * BS)
            enc = tok(a[sl], b[sl], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(y[sl]).to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            run = 0.98 * run + 0.02 * loss.item() if k else loss.item()
            if k % 2000 == 0:
                el = time.time() - t0
                print(f"  step {k}/{steps} loss {run:.4f} {(k + 1) * BS / max(el, 1e-6):.0f} pairs/s "
                      f"eta {el / (k + 1) * (steps - k - 1) / 60:.0f} min", flush=True)
        model.save_pretrained(fold_dir(f))
        tok.save_pretrained(fold_dir(f))
        del model, opt


@torch.no_grad()
def score(models, tok, df):
    """Logit of each model, [n_models, n]; pairs sorted by length so batches carry little padding."""
    order = np.argsort((df["a"].str.len_chars() + df["b"].str.len_chars()).to_numpy())
    a, b = df["a"].to_list(), df["b"].to_list()
    out = np.empty((len(models), df.height), dtype=np.float32)
    for k in range(0, df.height, BS_INF):
        idx = order[k:k + BS_INF]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt").to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            for g, m in enumerate(models):
                out[g, idx] = m(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out


def predict_all():
    tok = AutoTokenizer.from_pretrained(fold_dir(0))
    models = [AutoModelForSequenceClassification.from_pretrained(fold_dir(f)).to(DEV).half().eval() for f in (0, 1)]
    jobs = [("train", j, [models[1 - f]]) for f in (0, 1) for j in FOLDS[f]] + [("test", j, models) for j in range(BUCKETS)]
    for split, j, ms in jobs:
        if done(ce_path(split, j)):
            continue
        with timer(f"ce predict {split} {j}"):
            df = with_text(uncertain(split, j), split)
            t0 = time.time()
            ce = score(ms, tok, df)
            print(f"  {df.height:,} pairs, {df.height / max(time.time() - t0, 1e-6):.0f} pairs/s", flush=True)
            tmp = ce_path(split, j).with_suffix(".tmp")   # write + rename: a synced copy is never half-written
            o = df.select("s1", "id").with_columns(pl.Series("ce", ce.mean(0)))
            if len(ms) == 2:   # test: per-model logits, so stacked booster g can read model 1-g (see s4_train doc)
                o = o.with_columns(pl.Series("ce_m0", ce[0]), pl.Series("ce_m1", ce[1]))
            o.write_parquet(tmp)
            os.replace(tmp, ce_path(split, j))
            del df


def count():
    for split in ("train", "test"):
        n = sum(uncertain(split, j).select(pl.len()).collect().item() for j in range(BUCKETS))
        print(f"{split}: {n:,} uncertain pairs in [{LO}, {HI}]", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("count", "all"):
        count()
    if what in ("train", "all"):
        train_fold(0)
        train_fold(1)
    if what in ("predict", "all"):
        predict_all()
