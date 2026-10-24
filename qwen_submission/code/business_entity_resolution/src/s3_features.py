"""Stage 3: pair features, one S1 bucket at a time. Out: {split}_feat_{j}.parquet (+ label for train)."""
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import WORK, done, timer
from s2_block import BUCKETS, cand_path

STR = [  # (feature, left col, scorer)
    ("core_tset", "core", fuzz.token_set_ratio),
    ("core_tsort", "core", fuzz.token_sort_ratio),
    ("core_ratio", "core", fuzz.ratio),
    ("core_partial", "core", fuzz.partial_ratio),
    ("name_ratio", "name_n", fuzz.ratio),
    ("addr_tset", "addr_n", fuzz.token_set_ratio),
    ("addr_tsort", "addr_n", fuzz.token_sort_ratio),
    ("addr_partial", "addr_n", fuzz.partial_ratio),
]


def feat_path(split, j):
    return WORK / f"{split}_feat_{j}.parquet"


def features(split, j):
    n = pl.scan_parquet(WORK / f"{split}_norm.parquet").select("id", "src", "name_n", "core", "addr_n", "nums")
    x = (pl.scan_parquet(cand_path(split, j))
         .join(n.drop("src"), left_on="s1", right_on="id")
         .join(n, on="id", suffix="_2").collect())
    cols = {}
    for f, c, scorer in STR:
        cols[f] = cpdist(x[c].to_list(), x[c + "_2"].to_list(), scorer=scorer, workers=-1, dtype=np.uint8)
    num_i = x["nums"].list.set_intersection(x["nums_2"]).list.len()
    num_u = x["nums"].list.set_union(x["nums_2"]).list.len()
    out = x.select(
        "s1", "id", pl.col("src").cast(pl.Int8), "nk0", "nk1", "nk2", "nk3", "nk4", "sc", "rk", "ncand_raw",
        pl.col("core").str.len_chars().cast(pl.Int16).alias("len_core"),
        pl.col("core_2").str.len_chars().cast(pl.Int16).alias("len_core2"),
        pl.col("addr_n_2").str.len_chars().cast(pl.Int16).alias("len_addr2"),
        pl.col("core_2").str.contains(r"[^\x00-\x7f]").alias("nonlatin2"),
        (pl.col("nums").list.first() == pl.col("nums_2").list.first()).fill_null(False).alias("num_first_eq"),
    ).with_columns(
        num_i.cast(pl.Int8).alias("num_inter"),
        (num_i / num_u.clip(lower_bound=1)).cast(pl.Float32).alias("num_jacc"),
        (num_u - num_i).cast(pl.Int8).alias("num_diff"),
        *[pl.Series(k, v) for k, v in cols.items()],
    )
    # relative-to-best features within the S1's candidate list
    out = out.with_columns(
        (pl.col("core_tset").cast(pl.Int16) + pl.col("addr_tset")).alias("pair_sum"),
    ).with_columns(
        (pl.col("pair_sum").max().over("s1") - pl.col("pair_sum")).alias("gap_s1_best"),
    )
    if split == "train":
        gt = pl.scan_parquet(WORK / "train_gt.parquet").filter(pl.col("s1") % BUCKETS == j).collect()
        out = out.join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "id"], how="left") \
                 .with_columns(pl.col("y").fill_null(0))
    return out


def main(splits):
    for split in splits:
        for j in range(BUCKETS):
            if done(feat_path(split, j)):
                continue
            with timer(f"features {split} {j}"):
                features(split, j).write_parquet(feat_path(split, j))


if __name__ == "__main__":
    main(sys.argv[1:] or ("train", "test"))
