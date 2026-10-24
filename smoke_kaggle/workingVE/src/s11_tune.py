"""Threshold tuning under test-like decoy density.
Test has ~18% fewer S1 per S2/S3 record than train (train 4.67 S2+S3 per S1, test 5.5-5.8), i.e. more look-alike records
whose true entity is not in S1 (decoys). Simulate that on train OOF predictions: drop DROP of the S1 entities (their true
S2/S3 records stay in the pool as decoys), re-run the global assignment, score macro F0.5 on the remaining S1.
Prints the curve per seed, picks the best threshold, and (with --write) writes the test submission at that threshold.
usage: s11_tune.py <stage> [--write]    (reads p{stage}{TAG}_train/test; LB check: v3 s1 t=0.625 -> 0.961, t=0.80 -> 0.963)"""
import json
import os
import sys

import numpy as np
import polars as pl

from common import OUT, TAG, WORK, f05, timer
from s2_block import BUCKETS
from s4_train import assign, p_path

DROP = float(os.environ.get("BER_DROP", 0.18))
SEEDS = (1, 2)
GRID = [round(x, 3) for x in np.arange(0.5, 0.981, 0.025)]


def main(stage, write):
    pred = pl.concat([pl.read_parquet(p_path(stage, "train", j), columns=["s1", "id", "p"]).filter(pl.col("p") >= GRID[0])
                      for j in range(BUCKETS)])
    truth = pl.read_parquet(WORK / "train_gt.parquet")
    s1 = (pl.scan_parquet(WORK / "train_norm2.parquet").filter(pl.col("src") == 1)
          .select(pl.col("id").alias("s1")).collect())
    base = json.load(open(WORK / f"val_stage{stage}{TAG}.json"))
    print(f"stage {stage}{TAG}: normal OOF best t={base['t']} F={base['f05']:.4f}", flush=True)
    curves = []
    for seed in SEEDS:
        with timer(f"seed {seed}"):
            keep = s1.filter(pl.struct("s1").hash(seed=seed) % 1000 >= int(DROP * 1000))
            pk = pred.join(keep, on="s1", how="semi")        # dropped S1 vanish; their records stay as decoys
            tk = truth.join(keep, on="s1", how="semi")
            curve = {t: f05(keep, assign(pk, t), tk) for t in GRID}
            curves.append(curve)
            b = max(curve, key=curve.get)
            print(f"  seed {seed}: kept {keep.height:,} S1 | best t={b} F={curve[b]:.4f} | "
                  + " ".join(f"{t}:{curve[t]:.4f}" for t in GRID[::2]), flush=True)
    mean = {t: float(np.mean([c[t] for c in curves])) for t in GRID}
    bt = max(mean, key=mean.get)
    print(f"DECOY-SIM best t={bt} (mean F {mean[bt]:.4f}); normal-OOF t={base['t']}", flush=True)
    json.dump({"t": bt, "drop": DROP, "curve": mean, "normal_t": base["t"]},
              open(WORK / f"val_decoy{stage}{TAG}.json", "w"))
    if write:
        from s6_write import to_tsv
        out = OUT.parent / (OUT.name + f"_t{int(round(bt * 100))}")
        out.mkdir(parents=True, exist_ok=True)
        p = pl.concat([pl.read_parquet(p_path(stage, "test", j), columns=["s1", "id", "p"]).filter(pl.col("p") >= bt)
                       for j in range(BUCKETS)])
        s1_all = (pl.scan_parquet(WORK / "test_norm2.parquet").filter(pl.col("src") == 1)
                  .select(pl.col("id").alias("s1")).sort("s1").collect())
        m = assign(p, bt)
        to_tsv(s1_all, m, "matched_entity_ids", out / "matching_results.tsv")
        print(f"wrote {out}/matching_results.tsv: t={bt}, matched pairs {m.height:,}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), "--write" in sys.argv)
