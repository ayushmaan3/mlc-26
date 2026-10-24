"""Stage 9: per-S1 expected-F0.5 decoding (replaces one global threshold).
Macro F0.5 is averaged per S1, so the best prediction differs per S1: after the global assignment (each id -> its
best S1), sort that S1's candidates by p and keep the top-k that maximises
    E[F0.5 | k] ~= 1.25 * sum_{i<=k} q_i / (0.25 * sum_i q_i + k)     (k >= 1),   q = calibrated p
and predict nothing when P(no true match) = prod_i (1 - q_i) beats every k (singletons score 1.0 only when empty).
Calibration q = p ** g plus a floor f (pairs below f are ignored) are tuned on train OOF, never on the LB.
usage: s9_decode.py <stage>   -> writes WORK/val_decode{stage}.json and OUT(_ef) submission files."""
import json
import os
import sys

import numpy as np
import polars as pl

from common import OUT, TAG, WORK, f05, timer
from s2_block import BUCKETS
from s4_train import p_path
from s6_write import to_tsv

GAMMAS = [1.0, 1.25, 1.5]
FLOORS = [0.05, 0.15, 0.3]   # + the tuned global threshold t (added in main), so the search includes "t, then prune"


def assigned(pred, floor):
    """Global assignment: each id kept only for its best S1, and only if p >= floor."""
    return pred.filter(pl.col("p") >= floor).sort("p", descending=True).unique("id", keep="first")


def decode(pa, g):
    """pa: assigned pairs [s1, id, p]. Returns the kept pairs [s1, id]."""
    x = (pa.with_columns((pl.col("p").cast(pl.Float64) ** g).clip(1e-6, 1 - 1e-6).alias("q"))
         .sort(["s1", "q"], descending=[False, True])
         .with_columns(pl.col("q").cum_sum().over("s1").alias("cq"),
                       pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                       pl.col("q").sum().over("s1").alias("sq"),
                       (1 - pl.col("q")).log().sum().over("s1").exp().alias("p_empty")))
    x = x.with_columns((1.25 * pl.col("cq") / (0.25 * pl.col("sq") + pl.col("k"))).alias("ef"))
    best = x.group_by("s1").agg(pl.col("ef").max().alias("ef_best"),
                                pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"),
                                pl.col("p_empty").first())
    keep = best.filter(pl.col("ef_best") > pl.col("p_empty")).select("s1", "k_best")
    return x.join(keep, on="s1").filter(pl.col("k") <= pl.col("k_best")).select("s1", "id")


def main(stage):
    pred = pl.concat([pl.read_parquet(p_path(stage, "train", j)).filter(pl.col("p") >= min(FLOORS))
                      for j in range(BUCKETS)])
    truth = pl.read_parquet(WORK / "train_gt.parquet")
    s1 = (pl.scan_parquet(WORK / "train_norm2.parquet").filter(pl.col("src") == 1)
          .select(pl.col("id").alias("s1"), "country").collect())
    base = json.load(open(WORK / f"val_stage{stage}{TAG}.json"))
    print(f"stage {stage} global threshold: F={base['f05']:.5f} at t={base['t']}", flush=True)
    res = {}
    with timer("grid"):
        for f in FLOORS + [base["t"]]:
            pa = assigned(pred, f)
            for g in GAMMAS:
                res[(f, g)] = f05(s1, decode(pa, g), truth)
                print(f"  floor {f} gamma {g}: F={res[(f, g)]:.5f}", flush=True)
    (bf, bg), bF = max(res.items(), key=lambda kv: kv[1])
    d = bF - base["f05"]
    print(f"BEST expected-F decode: floor {bf} gamma {bg} F={bF:.5f}  (global threshold {base['f05']:.5f}, "
          f"delta {d:+.5f})", flush=True)
    print(">>> VERDICT: " + ("SUBMIT the _ef file (better on train OOF)" if d > 0.0005 else
                             "keep the global-threshold file (_ef is not better)"), flush=True)
    json.dump({"floor": bf, "gamma": bg, "f05": bF, "global_f05": base["f05"],
               "grid": {f"{k[0]}_{k[1]}": v for k, v in res.items()}}, open(WORK / f"val_decode{stage}{TAG}.json", "w"))
    del pred
    with timer("write test"):
        out = OUT.parent / (OUT.name + "_ef")
        out.mkdir(parents=True, exist_ok=True)
        tp = pl.concat([pl.read_parquet(p_path(stage, "test", j)) for j in range(BUCKETS)])
        s1_all = (pl.scan_parquet(WORK / "test_norm2.parquet").filter(pl.col("src") == 1)
                  .select(pl.col("id").alias("s1")).sort("s1").collect())
        m = decode(assigned(tp.filter(pl.col("p") >= bf), bf), bg)
        to_tsv(s1_all, m, "matched_entity_ids", out / "matching_results.tsv")
        to_tsv(s1_all, tp.select("s1", "id"), "candidate_entity_ids", out / "candidate_pairs.tsv")
        print("matched pairs", m.height, "S1 with >=1 match", m["s1"].n_unique(), "of", s1_all.height, "->", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
