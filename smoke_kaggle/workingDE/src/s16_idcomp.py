"""Per-record competition features, computed from similarities only (no model scores -> identical behaviour on train/test).
Half of all S1 names are shared by several S1 businesses, so for one S2/S3 record the model must pick WHICH same-named S1
has the best address. Each pair is otherwise scored in isolation. Here, for every pair (s1, id) of the v3 union:
  idc_n        : number of S1 candidates of this record (id)
  idc_rk_ps / idc_gap_ps   : rank of pair_sum (name+address fuzzy score) among the record's candidates / gap to the best rival
  idc_rk_ad / idc_gap_ad   : same for addr_tset
  idc_rk_em / idc_gap_em   : same for emb_cos
  idc_n_name90 : how many of the record's candidates have core_tset >= 90 (near-identical names = ambiguity)
  s1c_n_name90 : how many candidates of this S1 have core_tset >= 90
Needs all 8 feat3 buckets of a split (a record's candidates span buckets). Out: idc_{split}_{j}.parquet.
usage: s16_idcomp.py <split>"""
import os
import sys

import polars as pl

from common import WORK, done, timer
from s2_block import BUCKETS

COLS = ["s1", "id", "pair_sum", "addr_tset", "emb_cos", "core_tset"]


def idc_path(split, j):
    return WORK / f"idc_{split}_{j}.parquet"


def feat3(split, j):
    return WORK / f"{split}_feat3_{j}.parquet"


def build(split):
    if all(done(idc_path(split, j)) for j in range(BUCKETS)):
        return
    with timer(f"idcomp {split}"):
        x = pl.concat([pl.read_parquet(feat3(split, j), columns=COLS) for j in range(BUCKETS)])
        g = (x.group_by("id").agg(pl.len().cast(pl.Int16).alias("idc_n"),
                                  pl.col("pair_sum").max().alias("mx_ps"), pl.col("addr_tset").max().alias("mx_ad"),
                                  pl.col("emb_cos").max().alias("mx_em"),
                                  (pl.col("core_tset") >= 90).sum().cast(pl.Int16).alias("idc_n_name90")))
        s = x.group_by("s1").agg((pl.col("core_tset") >= 90).sum().cast(pl.Int16).alias("s1c_n_name90"))
        x = x.join(g, on="id", how="left").join(s, on="s1", how="left")
        x = x.with_columns(
            pl.col("pair_sum").rank("min", descending=True).over("id").cast(pl.Int16).alias("idc_rk_ps"),
            pl.col("addr_tset").rank("min", descending=True).over("id").cast(pl.Int16).alias("idc_rk_ad"),
            pl.col("emb_cos").rank("min", descending=True).over("id").cast(pl.Int16).alias("idc_rk_em"),
            (pl.col("mx_ps") - pl.col("pair_sum")).cast(pl.Int16).alias("idc_gap_ps"),
            (pl.col("mx_ad").cast(pl.Int16) - pl.col("addr_tset").cast(pl.Int16)).alias("idc_gap_ad"),
            (pl.col("mx_em") - pl.col("emb_cos")).cast(pl.Float32).alias("idc_gap_em"),
        ).select("s1", "id", "idc_n", "idc_rk_ps", "idc_gap_ps", "idc_rk_ad", "idc_gap_ad", "idc_rk_em", "idc_gap_em",
                 "idc_n_name90", "s1c_n_name90")
        for j in range(BUCKETS):
            tmp = idc_path(split, j).with_suffix(".tmp")
            x.filter(pl.col("s1") % BUCKETS == j).write_parquet(tmp)
            os.replace(tmp, idc_path(split, j))
        del x


if __name__ == "__main__":
    build(sys.argv[1])
