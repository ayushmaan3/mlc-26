"""Stage 6: final decode on test + write both submission TSVs.
matching_results: global assignment (each id -> best S1) with p >= threshold tuned on train OOF.
candidate_pairs: every pair the model scored (the exact inference set)."""
import json
import sys

import polars as pl

from common import OUT, TAG, WORK, decode_ids, timer
from s2_block import BUCKETS
from s4_train import assign, p_path


def to_tsv(s1_all, pairs, col, path):
    g = pairs.sort("s1", "id").group_by("s1", maintain_order=True).agg(decode_ids("id").str.join(",").alias(col))
    (s1_all.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
     .select(decode_ids("s1").alias("source1_entity_id"), col)
     .write_csv(path, separator="\t", quote_style="never"))


def main(stage):
    t = json.load(open(WORK / f"val_stage{stage}{TAG}.json"))["t"]
    OUT.mkdir(parents=True, exist_ok=True)
    s1_all = (pl.scan_parquet(WORK / "test_norm2.parquet").filter(pl.col("src") == 1)
              .select(pl.col("id").alias("s1")).sort("s1").collect())
    with timer(f"write matches stage {stage} t={t}"):
        p = pl.concat([pl.read_parquet(p_path(stage, "test", j)).filter(pl.col("p") >= t) for j in range(BUCKETS)])
        match = assign(p, t)
        to_tsv(s1_all, match, "matched_entity_ids", OUT / "matching_results.tsv")
        print("matched pairs", match.height, "S1 with >=1 match", match["s1"].n_unique(), "of", s1_all.height)
        del p, match
    with timer("write candidates"):
        cand = pl.concat([pl.read_parquet(p_path(stage, "test", j), columns=["s1", "id"]) for j in range(BUCKETS)])
        to_tsv(s1_all, cand, "candidate_entity_ids", OUT / "candidate_pairs.tsv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
