"""Blocking v3: key candidates (cand2) + embedding-kNN candidates (knn@K from e1_emb.py).
Measured on train: recall 0.9503 (keys) -> 0.9784 (keys + kNN@20), +12.8 pairs/S1.
  cos   : embedding cosine for every pair of the union -> embcos3_{split}_{j}.parquet [s1, id, emb_cos]
          (pairs sorted by embedding row, so the memmap is read nearly sequentially; S1 rows held in RAM)
  build : feat3_{split}_{j} = feat2 rows (key pairs, reused as-is) + s3 features of the NEW pairs only,
          + emb_cos, erk (kNN rank, null if not a kNN pair), in_keys, in_knn, and per-S1 windows recomputed
          over the union. Processed in S1 quarters (windows are per S1, so quarters are exact).
Then run s4/s5/s6/s8/s9 with BER_FEAT=feat3 BER_TAG=_v3 (v2 files are never overwritten).
usage: s10_v3.py cos <split> | s10_v3.py build <split> [bucket]"""
import os
import sys

import numpy as np
import polars as pl
import torch

from common import WORK, done, timer
from s2_block import BUCKETS, cand_path
import s3_features as s3

K3 = int(os.environ.get("BER_KNN_K3", 20))
QUART = 4
DIM = 384
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def knn_path(split, j):
    return WORK / f"knn_{split}_{j}.parquet"


def embcos3_path(split, j):
    return WORK / f"embcos3_{split}_{j}.parquet"


def feat2_path(split, j):
    return WORK / f"{split}_feat2_{j}.parquet"


def feat3_path(split, j):
    return WORK / f"{split}_feat3_{j}.parquet"


def knnft_path(split, j):
    return WORK / f"knnft_{split}_{j}.parquet"


def knn_pairs(split, j):
    """[s1, id] of kNN@K3, plus kNN@K3 of the fine-tuned e5 (knnft_*, s17) when present. The fine-tuned model only
    adds candidates; every feature (emb_cos, erk, windows) still comes from the original e5, so train/test stay alike."""
    k = [pl.read_parquet(knn_path(split, j)).filter(pl.col("erk") <= K3).select("s1", "id")]
    if knnft_path(split, j).exists():
        k.append(pl.read_parquet(knnft_path(split, j)).filter(pl.col("erk") <= K3).select("s1", "id"))
    return pl.concat(k).unique()


def union_pairs(split, j):
    """[s1, id] of cand2 ∪ kNN@K3 (∪ fine-tuned kNN@K3) for bucket j."""
    c = pl.read_parquet(cand_path(split, j), columns=["s1", "id"])
    return pl.concat([c, knn_pairs(split, j)]).unique()


NA = 3   # anchors per S1 for the cluster-support features


def cos(split):
    """emb_cos = cosine(S1, candidate) for every union pair, plus CLUSTER SUPPORT: in train an S1 usually has 2-5
    duplicates in S2 and in S3, and true duplicates resemble each other. Anchors = each S1's NA candidates with the
    highest emb_cos; for every candidate x: sup_max / sup_mean = max / mean cosine(x, anchor) over the S1's anchors
    other than x itself, sup_n90 = how many of them have cosine > 0.9. No learned scores are used (train/test consistent)."""
    ids = np.load(WORK / f"{split}_emb_ids.npy")
    E = np.memmap(WORK / f"{split}_emb.f16", dtype=np.float16, mode="r", shape=(len(ids), DIM))
    row = pl.DataFrame({"id": ids}).with_row_index("row")
    for j in range(BUCKETS):
        if done(embcos3_path(split, j)):
            continue
        with timer(f"cos3 {split} {j}"):
            u = (union_pairs(split, j).join(row.rename({"id": "s1", "row": "r1"}), on="s1")
                 .join(row.rename({"row": "r2"}), on="id").sort("r2"))
            s1_rows = np.unique(u["r1"].to_numpy())
            S = torch.from_numpy(np.ascontiguousarray(E[s1_rows])).to(DEV).float()   # this bucket's S1 embeddings
            loc = np.searchsorted(s1_rows, u["r1"].to_numpy()).astype(np.int64)
            r2 = u["r2"].to_numpy().astype(np.int64)   # polars row index is UInt32; torch needs int64
            out = np.empty(u.height, dtype=np.float32)
            for i in range(0, u.height, 500_000):
                b = torch.from_numpy(np.ascontiguousarray(E[r2[i:i + 500_000]])).to(DEV).float()  # sorted rows
                a = S[torch.from_numpy(loc[i:i + 500_000]).to(DEV)]
                out[i:i + 500_000] = (a * b).sum(1).cpu().numpy()
            del S
            u = u.with_columns(pl.Series("emb_cos", out), pl.Series("loc", loc))
            # anchors: top-NA candidates per S1 by emb_cos -> A_idx[s1 local index, k] = index into Aemb (-1 = none)
            anc = (u.select("loc", "r2", "emb_cos").sort(["loc", "emb_cos"], descending=[False, True])
                   .group_by("loc", maintain_order=True).head(NA)
                   .with_columns(pl.int_range(pl.len()).over("loc").alias("k")))
            arows = np.unique(anc["r2"].to_numpy())
            Aemb = torch.from_numpy(np.ascontiguousarray(E[arows])).to(DEV).float()
            A_idx = np.full((len(s1_rows), NA), -1, dtype=np.int64)
            A_row = np.full((len(s1_rows), NA), -1, dtype=np.int64)
            A_idx[anc["loc"].to_numpy(), anc["k"].to_numpy()] = np.searchsorted(arows, anc["r2"].to_numpy())
            A_row[anc["loc"].to_numpy(), anc["k"].to_numpy()] = anc["r2"].to_numpy().astype(np.int64)
            A_idx_t, A_row_t = torch.from_numpy(A_idx).to(DEV), torch.from_numpy(A_row).to(DEV)
            smax = np.empty(u.height, dtype=np.float32); smean = np.empty(u.height, dtype=np.float32)
            sn90 = np.empty(u.height, dtype=np.int8)
            CH = 250_000
            for i in range(0, u.height, CH):
                bx = torch.from_numpy(np.ascontiguousarray(E[r2[i:i + CH]])).to(DEV).float()
                lt = torch.from_numpy(loc[i:i + CH]).to(DEV)
                ai, ar = A_idx_t[lt], A_row_t[lt]                                  # [n, NA]
                sims = (Aemb[ai.clamp(min=0)] * bx[:, None, :]).sum(-1)            # [n, NA]
                valid = (ai >= 0) & (ar != torch.from_numpy(r2[i:i + CH]).to(DEV)[:, None])
                nv = valid.sum(1)
                smax[i:i + CH] = torch.where(nv > 0, sims.masked_fill(~valid, -2.0).max(1).values,
                                             torch.full_like(nv, float("nan"), dtype=torch.float32)).cpu().numpy()
                smean[i:i + CH] = torch.where(nv > 0, (sims * valid).sum(1) / nv.clamp(min=1),
                                              torch.full_like(nv, float("nan"), dtype=torch.float32)).cpu().numpy()
                sn90[i:i + CH] = ((sims > 0.9) & valid).sum(1).to(torch.int8).cpu().numpy()
            del Aemb
            tmp = embcos3_path(split, j).with_suffix(".tmp")
            (u.select("s1", "id", "emb_cos")
             .with_columns(pl.Series("sup_max", smax), pl.Series("sup_mean", smean), pl.Series("sup_n90", sn90))
             .write_parquet(tmp))
            os.replace(tmp, embcos3_path(split, j))
            del u
            torch.cuda.empty_cache()   # long runs fragment GPU memory (OOM seen on a T4 in cos test)


def new_pairs(split, j, c2):
    """kNN pairs not already in cand2, shaped like cand2 rows (no shared keys: nk*=0, sc=0, rk=99)."""
    k = knn_pairs(split, j).join(c2.select("s1", "id"), on=["s1", "id"], how="anti")
    nraw = c2.group_by("s1").agg(pl.col("ncand_raw").first())
    nk = [c for c in c2.columns if c.startswith("nk")]
    new = (k.join(nraw, on="s1", how="left")
           .with_columns(*[pl.lit(0).alias(c) for c in nk], pl.lit(0).alias("sc"), pl.lit(99).alias("rk"),
                         pl.col("ncand_raw").fill_null(0)))
    return new.select(c2.columns).cast(dict(c2.schema))


def windows(df):
    """Per-S1 relative features over the whole union (the feat2 values only saw key pairs)."""
    return df.with_columns(
        (pl.col("pair_sum").max().over("s1") - pl.col("pair_sum")).alias("gap_s1_best"),
        pl.col("pair_sum").rank("min", descending=True).over("s1").cast(pl.Int16).alias("rk_s1_ps"),
        (pl.col("pair_sum2").max().over("s1") - pl.col("pair_sum2")).alias("gap_s1_best2"),
        pl.col("emb_cos").rank("min", descending=True).over("s1").cast(pl.Int16).alias("emb_rk"),
        (pl.col("emb_cos").max().over("s1") - pl.col("emb_cos")).alias("emb_gap"),
        pl.len().over("s1").cast(pl.Int16).alias("ncand3"),
    )


def build(split, buckets=range(BUCKETS)):
    idf = None
    for j in buckets:
        if done(feat3_path(split, j)):
            continue
        with timer(f"feat3 {split} {j}"):
            c2 = pl.read_parquet(cand_path(split, j))
            new = new_pairs(split, j, c2)
            del c2
            if idf is None:
                idf = {col: pl.read_parquet(s3.idf_path(split, col)).with_columns(pl.col("country").cast(pl.Utf8))
                       for col in ("ch", "ah")}
            n = s3.load_records(split, pl.concat([new["s1"], new["id"]]).unique().to_frame("id"))
            gt = None
            if split == "train":
                gt = (pl.scan_parquet(WORK / "train_gt.parquet").filter(pl.col("s1") % BUCKETS == j)
                      .with_columns(pl.lit(1, pl.Int8).alias("y")).collect())
            knn = (pl.read_parquet(knn_path(split, j)).filter(pl.col("erk") <= K3)
                   .select("s1", "id", pl.col("erk").cast(pl.Int16)))
            ec = pl.read_parquet(embcos3_path(split, j))
            print(f"  new pairs {new.height:,} ({s3.rss()})", flush=True)
            parts = []
            for q in range(QUART):
                piece = (pl.col("s1") // BUCKETS) % QUART == q
                old = pl.read_parquet(feat2_path(split, j)).filter(piece)
                nq = new.filter(piece)
                fn = s3.finish_chunk(s3.features_chunk(nq, n, idf), gt) if nq.height else old.head(0)
                u = pl.concat([old.with_columns(pl.lit(1, pl.Int8).alias("in_keys")),
                               fn.select(old.columns).with_columns(pl.lit(0, pl.Int8).alias("in_keys"))],
                              how="vertical_relaxed")
                u = (u.join(ec, on=["s1", "id"], how="left").join(knn, on=["s1", "id"], how="left")
                     .with_columns(pl.col("erk").is_not_null().cast(pl.Int8).alias("in_knn")))
                pf = WORK / f"_tmp3_{split}_{j}_{q}.parquet"
                windows(u).write_parquet(pf)
                parts.append(pf)
                del old, nq, fn, u
                print(f"  quarter {q + 1}/{QUART} {s3.rss()}", flush=True)
            del n, new, knn, ec
            tmp = feat3_path(split, j).with_suffix(".tmp")
            pl.scan_parquet(parts).sink_parquet(tmp)
            os.replace(tmp, feat3_path(split, j))
            for pf in parts:
                pf.unlink()


def report():
    """Recall of feat3 train vs ground truth (should match the kNN@K3 union recall, ~0.978 for K=20)."""
    gt = pl.read_parquet(WORK / "train_gt.parquet")
    hit = tot = rows = 0
    for j in range(BUCKETS):
        f = pl.read_parquet(feat3_path("train", j), columns=["s1", "id", "y"])
        g = gt.filter(pl.col("s1") % BUCKETS == j)
        hit += int(f["y"].sum()); tot += g.height; rows += f.height
    print(f"feat3 train: {rows:,} pairs, positives {hit:,} / {tot:,} = recall {hit / tot:.4f}", flush=True)


if __name__ == "__main__":
    step, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "train")
    if step == "cos":
        cos(split)
    elif step == "build":
        build(split, [int(sys.argv[3])] if len(sys.argv) > 3 else range(BUCKETS))
    elif step == "report":
        report()
