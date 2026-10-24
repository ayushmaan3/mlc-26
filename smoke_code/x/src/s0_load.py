"""Stage 0: TSV -> parquet, once. Records: [id:int64, src:int8, name, addr, country]. GT: [s1, id] pairs."""
import polars as pl

from common import DATA, SPLITS, WORK, done, encode_ids, timer


def read_tsv(path):
    # quote_char=None: fields contain raw quotes; tabs are the only delimiter
    return pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)


def main():
    for split in SPLITS:
        out = WORK / f"{split}_records.parquet"
        if done(out):
            continue
        with timer(f"load {split}"):
            lf = pl.concat([read_tsv(DATA / split / f"{split}_source{k}.tsv") for k in (1, 2, 3)])
            df = lf.select(
                encode_ids("entity_id").alias("id"),
                pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("src"),
                pl.col("business_name").fill_null("").alias("name"),
                pl.col("business_address").fill_null("").alias("addr"),
                pl.col("country").fill_null("").cast(pl.Categorical).alias("country"),
            ).collect()
            assert df["id"].is_unique().all(), "duplicate ids"
            print(split, df.group_by("src").len().sort("src").rows())
            df.write_parquet(out)
            del df

    out = WORK / "train_gt.parquet"
    if not done(out):
        with timer("load gt"):
            gt = (read_tsv(DATA / "train" / "train_ground_truth.tsv")
                  .select(encode_ids("source1_entity_id").alias("s1"),
                          pl.col("matched_entity_ids").fill_null("").str.split(",").alias("id"))
                  .explode("id").filter(pl.col("id") != "")
                  .select("s1", encode_ids("id").alias("id")).collect())
            print("gt pairs", gt.height)
            gt.write_parquet(out)


if __name__ == "__main__":
    main()
