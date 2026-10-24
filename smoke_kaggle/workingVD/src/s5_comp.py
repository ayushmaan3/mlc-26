"""Stage 5: competition features from stage-1 probabilities (train: OOF, test: fold-mean).
Each S2/S3 id belongs to at most one S1 and each S1 has a handful of true ids, so a pair's p relative to its
rivals (same id across ALL S1 buckets; same S1) is strong evidence. Out: comp_{split}_{j}.parquet."""
import sys

import polars as pl

from common import WORK, done, timer
from s2_block import BUCKETS
from s4_train import comp_path, p_path


def build(split, stage=1, variant=None):
    """variant (test only): build from p_m{variant} = the test predictions of stage-1 model `variant`."""
    if all(done(comp_path(split, j, variant)) for j in range(BUCKETS)):
        return
    pcol = "p" if variant is None else f"p_m{variant}"
    with timer(f"comp {split}" + ("" if variant is None else f" from model {variant}")):
        src = [pl.scan_parquet(p_path(stage, split, j)).select("s1", "id", pl.col(pcol).alias("p")) for j in range(BUCKETS)]
        # per id across every S1 bucket: best / second-best p, number of candidate S1s
        ids = (pl.concat(src).group_by("id")
               .agg(pl.col("p").top_k(2).alias("top"), pl.len().cast(pl.Int16).alias("id_n"),
                    (pl.col("p") > 0.1).sum().cast(pl.Int16).alias("id_n10"))
               .select("id", "id_n", "id_n10", pl.col("top").list.get(0).alias("id_max"),
                       pl.col("top").list.get(1, null_on_oob=True).fill_null(0.0).alias("id_2nd"))
               .collect())
        for j in range(BUCKETS):
            p = (pl.read_parquet(p_path(stage, split, j)).select("s1", "id", pl.col(pcol).alias("p"))
                 .join(ids, on="id", how="left"))
            top = pl.col("p") >= pl.col("id_max")
            (p.with_columns(
                top.alias("id_top"),
                pl.when(top).then(pl.col("id_2nd")).otherwise(pl.col("id_max")).alias("id_rival"),
                pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("s1_rank"),
                pl.col("p").max().over("s1").alias("s1_max"),
                pl.col("p").sum().over("s1").cast(pl.Float32).alias("s1_sum"),
                (pl.col("p") > 0.5).sum().over("s1").cast(pl.Int16).alias("s1_n50"))
             .with_columns((pl.col("p") - pl.col("id_rival")).alias("id_margin"),
                           (pl.col("s1_max") - pl.col("p")).alias("s1_gap"))
             .rename({"p": "p1"}).drop("id_max", "id_2nd")
             .write_parquet(comp_path(split, j, variant)))
            del p
        del ids


if __name__ == "__main__":
    for s in (sys.argv[1:] or ("train", "test")):
        if s == "test":
            build(s, variant=0)
            build(s, variant=1)
        else:
            build(s)
