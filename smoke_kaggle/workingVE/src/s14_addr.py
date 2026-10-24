"""Add-on address features for the house-number noise seen in the error budget: among rejected true pairs with an address,
72% have a different first house number and 45% of those are one digit edit away (1161 vs 161, 85 vs 15).
Per pair of the v3 union (cand2 + kNN@20, the same pairs as feat3):
  street_tset / street_ratio : address similarity with every token containing a digit removed
  num_best_lev               : smallest Levenshtein distance between any of the first 3 numbers on each side (9 = none)
  num_contain                : first numbers differ but one contains the other (161 in 1161)
  num_reldiff                : |a - b| / max(a, b) of the first numbers (NaN if missing)
Out: addr4_{split}_{j}.parquet [s1, id, ...] (model-independent, so untagged). Joined by s4_train as an extra (stages 3-5).
usage: s14_addr.py <split>"""
import os
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from common import WORK, done, timer
from s2_block import BUCKETS
import s3_features as s3
from s10_v3 import union_pairs   # same pairs as feat3 (incl. fine-tuned kNN pairs when present)

SUB = 8


def addr_path(split, j):
    return WORK / f"addr4_{split}_{j}.parquet"




def chunk_feats(x):
    street = lambda c: pl.col(c).str.replace_all(r"\S*\d\S*", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    x = x.with_columns(street("addr_n").alias("st1"), street("addr_n_2").alias("st2"))
    st1, st2 = x["st1"].to_list(), x["st2"].to_list()
    feats = {"street_tset": cpdist(st1, st2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8),
             "street_ratio": cpdist(st1, st2, scorer=fuzz.ratio, workers=-1, dtype=np.uint8)}
    best = np.full(x.height, 9, dtype=np.int8)
    for a in range(3):
        na = x["nums"].list.get(a, null_on_oob=True)
        for b in range(3):
            nb = x["nums_2"].list.get(b, null_on_oob=True)
            ok = (na.is_not_null() & nb.is_not_null()).to_numpy()
            if not ok.any():
                continue
            d = cpdist(na.fill_null("").to_list(), nb.fill_null("").to_list(), scorer=Levenshtein.distance, workers=-1,
                       dtype=np.int32)
            best = np.where(ok, np.minimum(best, d), best).astype(np.int8)
    feats["num_best_lev"] = best
    f1, f2 = pl.col("nums").list.first(), pl.col("nums_2").list.first()
    i1, i2 = f1.str.slice(0, 9).cast(pl.Int64, strict=False), f2.str.slice(0, 9).cast(pl.Int64, strict=False)
    out = x.select(
        "s1", "id",
        ((f1 != f2) & (f1.str.contains(f2, literal=True) | f2.str.contains(f1, literal=True))).fill_null(False).alias("num_contain"),
        ((i1 - i2).abs() / pl.max_horizontal(i1, i2).clip(lower_bound=1)).cast(pl.Float32).alias("num_reldiff"),
    )
    return out.with_columns(*[pl.Series(k, v) for k, v in feats.items()])


def build(split):
    for j in range(BUCKETS):
        if done(addr_path(split, j)):
            continue
        with timer(f"addr4 {split} {j}"):
            u = union_pairs(split, j)
            n = s3.load_records(split, pl.concat([u["s1"], u["id"]]).unique().to_frame("id")).select("id", "addr_n", "nums")
            parts = []
            for s in range(SUB):
                c = u.filter((pl.col("s1") // BUCKETS) % SUB == s)
                x = (c.join(n.rename({"id": "s1"}), on="s1").join(n, on="id", suffix="_2"))
                parts.append(chunk_feats(x))
                del c, x
            tmp = addr_path(split, j).with_suffix(".tmp")
            pl.concat(parts).write_parquet(tmp)
            os.replace(tmp, addr_path(split, j))
            del u, n, parts


if __name__ == "__main__":
    build(sys.argv[1])
