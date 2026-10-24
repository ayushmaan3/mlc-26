"""Run X: fold-wise cross-encoder on the UNCERTAIN pairs of the final model (0.5 <= p < 0.995), as a stacker feature (s22).
A perfect judge on these pairs would lift labelled OOF 0.98685 -> 0.99138; the CE (reads both records' raw text, multilingual,
incl. French) tries to get part of that. Reuses s8_ce (training loop / scoring) with other pair sources and output names:
  train pairs : run R's OOF p7 (oof_p7_ge05.parquet) in the band, fold f = S1 buckets FOLDS[f], labels from train_gt
  OOF scores  : fold f pairs scored by the model trained on the other fold; test: pairs with p_m0 or p_m1 in the band, mean of both
  out: ceband_{split}_{j}.parquet [s1, id, ce]
usage: s23_ceband.py train <f> | score"""
import sys

import polars as pl

from common import WORK
import s8_ce

LO, HI = 0.5, 0.995


def band(split, j):
    if split == "train":
        return (pl.scan_parquet(WORK / "oof_p7_ge05.parquet").filter((pl.col("s1") % 8 == j) & pl.col("p").is_between(LO, HI, closed="left"))
                .select("s1", "id"))
    return (pl.scan_parquet(WORK / f"p7_v3_test_{j}.parquet")
            .filter(pl.col("p_m0").is_between(LO, HI, closed="left") | pl.col("p_m1").is_between(LO, HI, closed="left")).select("s1", "id"))


def ceband_path(split, j):
    return WORK / f"ceband_{split}_{j}.parquet"


s8_ce.uncertain = band            # the pairs s8_ce trains on / scores
s8_ce.ce_path = ceband_path       # its outputs (never the dataset's ce_v3_* files)
s8_ce.fold_dir = lambda f: WORK / f"ceband_model_f{f}"

if __name__ == "__main__":
    if sys.argv[1] == "train":
        s8_ce.train_fold(int(sys.argv[2]))
    elif sys.argv[1] == "score":
        s8_ce.predict_all()
