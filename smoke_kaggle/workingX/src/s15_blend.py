"""Blend + write: average the probabilities of one or more stages (same pairs), tune the threshold on train OOF (macro F0.5,
global assignment), and write two test submissions:
  <OUT>/<name>/matching_results.tsv        at the OOF-best threshold
  <OUT>/<name>_fr90/matching_results.tsv   at t=0.80 (France 0.90)  - the stricter setting that won on the public LB
  <OUT>/<name>_t80/matching_results.tsv    at t=0.80 everywhere (France too): pairs with _fr90 to measure France's threshold
usage: s15_blend.py <name> <stage> [<stage> ...]      e.g.  s15_blend.py v3s5full 5    |   s15_blend.py blend56 5 6
       s15_blend.py write <name> <t> <t_france> <stage> [<stage> ...]   write one file from existing test p (no tuning)"""
import json
import os
import sys

import numpy as np
import polars as pl

from common import OUT, WORK, f05, timer
from s2_block import BUCKETS
from s4_train import assign, p_path
from s6_write import to_tsv

LOW = 0.4
GRID = [round(x, 3) for x in np.arange(0.5, 0.951, 0.025)]


def blended(stages, split, j, low=None):
    ps = [pl.read_parquet(p_path(st, split, j), columns=["s1", "id", "p"]).rename({"p": f"p{k}"}) for k, st in enumerate(stages)]
    b = ps[0]
    for q in ps[1:]:
        b = b.join(q, on=["s1", "id"], how="inner")
    b = b.select("s1", "id", pl.mean_horizontal([f"p{k}" for k in range(len(stages))]).alias("p"))
    return b if low is None else b.filter(pl.col("p") >= low)


def write(name, stages, rules):
    """rules: [(sub-folder name, None = OOF-best t (needs bt) or (t, t_france))]."""
    tp = pl.concat([blended(stages, "test", j, LOW) for j in range(BUCKETS)])
    s1_all = (pl.scan_parquet(WORK / "test_norm2.parquet").filter(pl.col("src") == 1)
              .select(pl.col("id").alias("s1")).sort("s1").collect())
    cty = pl.read_parquet(WORK / "test_records.parquet", columns=["id", "country"]).rename({"id": "s1"})
    for sub, rule in rules:
        if isinstance(rule, float):
            m = assign(tp, rule)
        else:
            q = tp.join(cty, on="s1").filter(pl.when(pl.col("country") == "France").then(pl.col("p") >= rule[1])
                                             .otherwise(pl.col("p") >= rule[0]))
            m = q.sort("p", descending=True).unique("id", keep="first").select("s1", "id")
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        to_tsv(s1_all, m, "matched_entity_ids", d / "matching_results.tsv")
        print(f"  wrote {d}/matching_results.tsv  matched pairs {m.height:,}", flush=True)
    return tp, s1_all


def main(name, stages):
    with timer(f"tune {name} {stages}"):
        pred = pl.concat([blended(stages, "train", j, LOW) for j in range(BUCKETS)])
        truth = pl.read_parquet(WORK / "train_gt.parquet")
        s1 = (pl.scan_parquet(WORK / "train_norm2.parquet").filter(pl.col("src") == 1)
              .select(pl.col("id").alias("s1"), "country").collect())
        curve = {t: f05(s1, assign(pred, t), truth) for t in GRID}
        bt = max(curve, key=curve.get)
        by = {c: f05(s1.filter(pl.col("country") == c), assign(pred, bt).join(s1.filter(pl.col("country") == c), on="s1", how="semi"),
                     truth.join(s1.filter(pl.col("country") == c), on="s1", how="semi")) for c in s1["country"].unique().to_list()}
        print(f"{name} {stages}: OOF best t={bt} F0.5={curve[bt]:.5f} | at 0.80: {curve[0.8]:.5f} | by country "
              + str({k: round(v, 4) for k, v in by.items()}), flush=True)
        json.dump({"stages": stages, "t": bt, "f05": curve[bt], "curve": curve, "by_country": by},
                  open(WORK / f"val_blend_{name}.json", "w"))
    with timer(f"write {name}"):
        tp, s1_all = write(name, stages, [(name, float(bt)), (name + "_fr90", (0.80, 0.90)), (name + "_t80", (0.80, 0.80))])
        cand = OUT / "candidates" / "candidate_pairs.tsv"
        if not cand.exists():
            cand.parent.mkdir(parents=True, exist_ok=True)
            allp = pl.concat([pl.read_parquet(p_path(stages[0], "test", j), columns=["s1", "id"]) for j in range(BUCKETS)])
            to_tsv(s1_all, allp, "candidate_entity_ids", cand)
            print(f"  wrote {cand}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "write":
        nm, t, tf = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
        with timer(f"write {nm} t={t} France={tf}"):
            write(nm, [int(x) for x in sys.argv[5:]], [(nm, (t, tf))])
    else:
        main(sys.argv[1], [int(x) for x in sys.argv[2:]])
