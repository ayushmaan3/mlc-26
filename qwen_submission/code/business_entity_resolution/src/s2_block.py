"""Stage 2: blocking. Keys (all within country):
see keys(). Blocks bigger than caps are dropped (generic keys). Out: [s1, id, nk0..nk4 = #shared keys per type] unique pairs."""
import sys

import polars as pl

from common import SPLITS, WORK, done, timer

CAP1, CAP23 = 20, 100   # max S1 / S2+S3 records sharing a key
PARTS = 16              # key-hash partitions -> bounded RAM


def rare(lf, col, n):
    """Per record keep the n rarest tokens of list column `col` that occur >= 2 times (typos are useless keys)."""
    t = lf.select("id", "country", pl.col(col).alias("t")).explode("t").drop_nulls("t").unique(["id", "t"])
    df = t.group_by("country", "t").len("df").filter(pl.col("df") >= 2)
    return (t.join(df, on=["country", "t"]).sort("id", "df", "t").group_by("id")
            .agg(pl.col("t").head(n).alias(col)))


def keys(lf):
    """kt: 0 core name | 1 number x rare addr token | 2 rare name tok x rare addr tok
           3 rare addr token pair | 4 rare name token pair"""
    lf = lf.with_columns(pl.col("core").str.split(" ").list.eval(
        pl.element().filter(pl.element().str.len_chars() >= 3)).alias("ntoks"))
    base = lf.select("id", "src", "country", "core", "nums").join(
        rare(lf, "atoks", 3), on="id", how="left").join(rare(lf, "ntoks", 3), on="id", how="left")

    def emit(kt, cols, frame):
        return frame.select("id", "src", pl.concat_str(["country", *cols], separator="|").hash().alias("k"),
                            pl.lit(kt, pl.Int8).alias("kt"))

    def pairs(col):
        a = base.select("id", "src", "country", pl.col(col).alias("a")).explode("a")
        b = base.select("id", pl.col(col).alias("b")).explode("b")
        return a.join(b, on="id").filter(pl.col("a") < pl.col("b"))

    return pl.concat([
        emit(0, ["core"], base.filter(pl.col("core") != "")),
        emit(1, ["nums", "atoks"], base.explode("nums").explode("atoks").drop_nulls(["nums", "atoks"])),
        emit(2, ["ntoks", "atoks"], base.explode("ntoks").explode("atoks").drop_nulls(["ntoks", "atoks"])),
        emit(3, ["a", "b"], pairs("atoks")),
        emit(4, ["a", "b"], pairs("ntoks").with_columns(pl.lit("n").alias("country2"))
             .with_columns(pl.concat_str(["country", "country2"]).alias("country"))),
    ]).unique(["id", "k"])


BUCKETS = 8   # S1-id buckets: every later stage works one bucket at a time
TOPK = 50     # candidates kept per S1 (ranked by weighted shared-key count)


def cand_path(split, j):
    return WORK / f"{split}_cand_{j}.parquet"


def block(split):
    if all(done(cand_path(split, j)) for j in range(BUCKETS)):
        return
    kf = WORK / f"{split}_keys.parquet"
    if not done(kf):
        with timer(f"keys {split}"):
            keys(pl.scan_parquet(WORK / f"{split}_norm.parquet").with_columns(pl.col("country").cast(pl.Utf8))
                 ).sink_parquet(kf)
    parts = []
    for p in range(PARTS):
        pf = WORK / f"{split}_cand_p{p}.parquet"
        if not done(pf):
            with timer(f"join {split} part {p}"):
                k = pl.scan_parquet(kf).filter(pl.col("k") % PARTS == p).collect()
                sz = k.group_by("k").agg((pl.col("src") == 1).sum().alias("n1"),
                                         (pl.col("src") != 1).sum().alias("n23"))
                good = sz.filter((pl.col("n1") >= 1) & (pl.col("n1") <= CAP1) &
                                 (pl.col("n23") >= 1) & (pl.col("n23") <= CAP23)).select("k")
                k = k.join(good, on="k", how="semi")
                a = k.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "k", "kt")
                b = k.filter(pl.col("src") != 1).select("id", "k")
                (a.join(b, on="k").group_by("s1", "id")
                 .agg(*[(pl.col("kt") == t).sum().cast(pl.Int16).alias(f"nk{t}") for t in range(5)])
                 .write_parquet(pf))
                del k, a, b
        parts.append(pf)
    # merge per S1 bucket (a pair can come from several key partitions) -> rank -> top-K
    for j in range(BUCKETS):
        if done(cand_path(split, j)):
            continue
        with timer(f"merge {split} bucket {j}"):
            c = (pl.scan_parquet(parts).filter(pl.col("s1") % BUCKETS == j).group_by("s1", "id")
                 .agg(pl.col(f"nk{t}").sum() for t in range(5)).collect())
            c = c.with_columns((3 * pl.col("nk0") + pl.col("nk1") + pl.col("nk2") + pl.col("nk3") + pl.col("nk4"))
                               .cast(pl.Int16).alias("sc"))
            c = c.with_columns(pl.col("sc").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("rk"),
                               pl.len().over("s1").cast(pl.Int32).alias("ncand_raw"))
            c.filter(pl.col("rk") <= TOPK).write_parquet(cand_path(split, j))
            del c
    for pf in parts:
        pf.unlink()


def recall_report():
    c = pl.scan_parquet([cand_path("train", j) for j in range(BUCKETS)])
    gt = pl.scan_parquet(WORK / "train_gt.parquet")
    n_s1 = pl.scan_parquet(WORK / "train_norm.parquet").filter(pl.col("src") == 1).select(pl.len()).collect().item()
    npairs = c.select(pl.len()).collect().item()
    hit = gt.join(c, on=["s1", "id"], how="inner").select(pl.len()).collect().item()
    ngt = gt.select(pl.len()).collect().item()
    print(f"pairs {npairs:,}  per S1 {npairs / n_s1:.1f}  recall {hit / ngt:.4f} ({hit:,}/{ngt:,})")


if __name__ == "__main__":
    for s in (sys.argv[1:] or SPLITS):
        block(s)
    recall_report()
