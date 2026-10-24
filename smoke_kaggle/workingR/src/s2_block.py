"""Stage 2: blocking. Keys (all within country): see keys(). Blocks bigger than caps are dropped (generic keys).
Out per S1 bucket: [s1, id, nk0..nk8 = #shared keys per type, sc, rk, ncand_raw], top-K per S1."""
import sys

import polars as pl

from common import SPLITS, WORK, done, timer

CAP1, CAP23 = 20, 100   # max S1 / S2+S3 records sharing a key
PARTS = 32              # key-hash partitions -> bounded RAM
NKT = 9
KW = [3, 1, 1, 1, 1, 3, 3, 1, 1]  # rank weight per key type


RARE = {"atoks": 3, "ntoks": 3, "stoks": 2}
KCHUNKS = 16  # key emission in id chunks -> bounded RAM


def ntoks(lf):
    return lf.with_columns(pl.col("core").str.split(" ").list.eval(
        pl.element().filter(pl.element().str.len_chars() >= 3)).alias("ntoks"))


def token_df(lf, col):
    """Document frequency per (country, token) over the whole split; tokens seen once are useless keys."""
    return (lf.select("id", "country", pl.col(col).list.unique().alias("t")).explode("t").drop_nulls("t")
            .group_by("country", "t").len("df").filter(pl.col("df") >= 2).collect())


def rare(lf, col, n, df):
    """Per record keep the n rarest tokens of list column `col`."""
    t = lf.select("id", "country", pl.col(col).list.unique().alias("t")).explode("t").drop_nulls("t")
    return (t.join(df.lazy(), on=["country", "t"]).sort("id", "df", "t").group_by("id")
            .agg(pl.col("t").head(n).alias(col)))


def keys(lf, dfs):
    """kt: 0 core name | 1 number x rare addr token | 2 rare name tok x rare addr tok | 3 rare addr token pair
           4 rare name token pair | 5 name skeleton | 6 name without spaces (web/handle names)
           7 rare skeleton tok x rare addr tok | 8 rare skeleton tok x number"""
    lf = ntoks(lf)
    base = lf.select("id", "src", "country", "core", "skel", "nosp", pl.col("nums").list.head(3))
    for col, n in RARE.items():
        base = base.join(rare(lf, col, n, dfs[col]), on="id", how="left")

    def emit(kt, cols, frame):
        return frame.select("id", "src", pl.concat_str(["country", pl.lit(str(kt)), *cols], separator="|")
                            .hash().alias("k"), pl.lit(kt, pl.Int8).alias("kt"))

    def pairs(col):
        a = base.select("id", "src", "country", pl.col(col).alias("a")).explode("a")
        b = base.select("id", pl.col(col).alias("b")).explode("b")
        return a.join(b, on="id").filter(pl.col("a") < pl.col("b"))

    def cross(c1, c2):
        return base.explode(c1).explode(c2).drop_nulls([c1, c2])

    return pl.concat([
        emit(0, ["core"], base.filter(pl.col("core") != "")),
        emit(1, ["nums", "atoks"], cross("nums", "atoks")),
        emit(2, ["ntoks", "atoks"], cross("ntoks", "atoks")),
        emit(3, ["a", "b"], pairs("atoks")),
        emit(4, ["a", "b"], pairs("ntoks")),
        emit(5, ["skel"], base.filter(pl.col("skel").str.len_chars() >= 4)),
        emit(6, ["nosp"], base.filter(pl.col("nosp").str.len_chars() >= 6)),
        emit(7, ["stoks", "atoks"], cross("stoks", "atoks")),
        emit(8, ["stoks", "nums"], cross("stoks", "nums")),
    ]).unique(["id", "k"])


BUCKETS = 8   # S1-id buckets: every later stage works one bucket at a time
TOPK = int(__import__("os").environ.get("BER_TOPK", 60))   # candidates kept per S1


def cand_path(split, j):
    return WORK / f"{split}_cand2_{j}.parquet"


def block(split):
    if all(done(cand_path(split, j)) for j in range(BUCKETS)):
        return
    kfs = [WORK / f"{split}_keys2_{c}.parquet" for c in range(KCHUNKS)]
    if not all(done(f) for f in kfs):
        lf = ntoks(pl.scan_parquet(WORK / f"{split}_norm2.parquet").with_columns(pl.col("country").cast(pl.Utf8)))
        with timer(f"token df {split}"):
            dfs = {col: token_df(lf, col) for col in RARE}
        for c, f in enumerate(kfs):
            if not done(f):
                with timer(f"keys {split} chunk {c}"):
                    keys(lf.filter(pl.col("id") % KCHUNKS == c), dfs).collect().write_parquet(f)
        del dfs
    kf = kfs
    parts = []
    for p in range(PARTS):
        pf = WORK / f"{split}_cand2_p{p}.parquet"
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
                 .agg(*[(pl.col("kt") == t).sum().cast(pl.Int8).alias(f"nk{t}") for t in range(NKT)])
                 .write_parquet(pf))
                del k, a, b
        parts.append(pf)
    # merge per S1 bucket (a pair can come from several key partitions) -> rank -> top-K
    for j in range(BUCKETS):
        if done(cand_path(split, j)):
            continue
        with timer(f"merge {split} bucket {j}"):
            c = (pl.scan_parquet(parts).filter(pl.col("s1") % BUCKETS == j).group_by("s1", "id")
                 .agg(pl.col(f"nk{t}").sum() for t in range(NKT)).collect())
            c = c.with_columns(pl.sum_horizontal([KW[t] * pl.col(f"nk{t}").cast(pl.Int16) for t in range(NKT)])
                               .cast(pl.Int16).alias("sc"))
            c = c.with_columns(pl.col("sc").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("rk"),
                               pl.len().over("s1").cast(pl.Int32).alias("ncand_raw"))
            c.filter(pl.col("rk") <= TOPK).write_parquet(cand_path(split, j))
            if split == "train":   # uncapped copy of a single bucket, to measure what top-K loses
                if j == 0:
                    c.select("s1", "id", "sc", "rk").write_parquet(WORK / "train_cand2_full0.parquet")
            del c
    for pf in parts:
        pf.unlink()


def recall_report():
    c = pl.scan_parquet([cand_path("train", j) for j in range(BUCKETS)])
    gt = pl.scan_parquet(WORK / "train_gt.parquet")
    n_s1 = pl.scan_parquet(WORK / "train_norm2.parquet").filter(pl.col("src") == 1).select(pl.len()).collect().item()
    npairs = c.select(pl.len()).collect().item()
    hit = gt.join(c, on=["s1", "id"], how="inner").select(pl.len()).collect().item()
    ngt = gt.select(pl.len()).collect().item()
    print(f"pairs {npairs:,}  per S1 {npairs / n_s1:.1f}  recall {hit / ngt:.4f} ({hit:,}/{ngt:,})")
    f = pl.read_parquet(WORK / "train_cand2_full0.parquet")
    g0 = gt.filter(pl.col("s1") % BUCKETS == 0).collect()
    h = g0.join(f, on=["s1", "id"])
    print("bucket0 uncapped recall", round(h.height / g0.height, 4),
          {k: round((h["rk"] <= k).sum() / g0.height, 4) for k in (10, 20, 30, 40, 50, 60)})


if __name__ == "__main__":
    for s in (sys.argv[1:] or SPLITS):
        block(s)
    recall_report()
