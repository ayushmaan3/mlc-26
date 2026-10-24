"""Error budget of a stage's OOF decisions (train). How much macro F0.5 would we gain by removing each error type alone?
  FP_single : predicted matches on S1 that have NO true match (singletons)
  FP_owned  : predicted id truly belongs to ANOTHER S1 (wrong assignment)
  FP_decoy  : predicted id matches nothing at all (look-alike decoy)
  FP_other  : the rest of the false pairs on matched S1
  FN_cand   : true pair in the candidates but not predicted (low p, or the id was assigned elsewhere)
  FN_block  : true pair not in the candidates at all
usage: s13_errors.py <stage>   (BER_TAG=_v3 for v3 files; needs train_gt, train_norm2, p{stage}{TAG}_train_*, knn/cand for FN_block)"""
import json
import sys

import polars as pl

from common import TAG, WORK, f05_scores
from s2_block import BUCKETS
from s4_train import assign, p_path


def main(stage):
    t = json.load(open(WORK / f"val_stage{stage}{TAG}.json"))["t"]
    s1 = (pl.scan_parquet(WORK / "train_norm2.parquet").filter(pl.col("src") == 1)
          .select(pl.col("id").alias("s1"), "country").collect())
    truth = pl.read_parquet(WORK / "train_gt.parquet")
    p = pl.concat([pl.read_parquet(p_path(stage, "train", j), columns=["s1", "id", "p"]) for j in range(BUCKETS)])
    pred = assign(p.filter(pl.col("p") >= t), t)
    cand = p.select("s1", "id")
    base = f05_scores(s1, pred, truth)["f"].mean()
    tp = pred.join(truth, on=["s1", "id"], how="semi")
    fp = pred.join(truth, on=["s1", "id"], how="anti")
    matched_s1 = truth.select("s1").unique()
    owner = truth.rename({"s1": "true_s1"})
    fp = fp.join(owner, on="id", how="left").with_columns(
        pl.when(~pl.col("s1").is_in(matched_s1["s1"].implode())).then(pl.lit("FP_single"))
        .when(pl.col("true_s1").is_not_null()).then(pl.lit("FP_owned"))
        .otherwise(pl.lit("FP_decoy")).alias("kind"))
    fn = truth.join(pred, on=["s1", "id"], how="anti")
    fn = pl.concat([fn.join(cand, on=["s1", "id"], how="semi").with_columns(pl.lit("FN_cand").alias("kind")),
                    fn.join(cand, on=["s1", "id"], how="anti").with_columns(pl.lit("FN_block").alias("kind"))])
    print(f"stage {stage}{TAG} t={t}: OOF macro F0.5 {base:.4f} | pairs: TP {tp.height:,} FP {fp.height:,} FN {fn.height:,}", flush=True)
    print("  pair precision %.4f  pair recall %.4f" % (tp.height / max(pred.height, 1), tp.height / truth.height), flush=True)
    print("  counts:", dict(sorted(fp.group_by("kind").len().rows() + fn.group_by("kind").len().rows())), flush=True)
    print("  macro-F0.5 gain if ONLY this error type were fixed:", flush=True)
    for k in ("FP_single", "FP_owned", "FP_decoy"):
        bad = fp.filter(pl.col("kind") == k).select("s1", "id")
        f = f05_scores(s1, pred.join(bad, on=["s1", "id"], how="anti"), truth)["f"].mean()
        print(f"    {k:10s} n={bad.height:>9,}  +{f - base:.4f}", flush=True)
    for k in ("FN_cand", "FN_block"):
        add = fn.filter(pl.col("kind") == k).select("s1", "id")
        f = f05_scores(s1, pl.concat([pred, add]), truth)["f"].mean()
        print(f"    {k:10s} n={add.height:>9,}  +{f - base:.4f}", flush=True)
    sc = f05_scores(s1, pred, truth).join(s1, on="s1")
    print("  by country:", {r[0]: round(r[1], 4) for r in sc.group_by("country").agg(pl.col("f").mean()).rows()}, flush=True)
    print("  singletons: share of S1 %.3f, F on them %.4f | matched S1: F %.4f" % (
        (sc["nt"] == 0).mean(), sc.filter(pl.col("nt") == 0)["f"].mean(), sc.filter(pl.col("nt") > 0)["f"].mean()), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
