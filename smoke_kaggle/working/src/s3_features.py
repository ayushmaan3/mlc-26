"""Stage 3: pair features, one S1 bucket per process (bounded RAM on a 12 GB box).
prep (once per split): {split}_norm3.parquet = norm2 cols + hashed token lists + nonascii flag + name frequencies,
                       {split}_idf_{ch,ah}.parquet = per-country IDF of name / address tokens.
bucket j: loads only the records its candidate pairs touch, computes features in sub-chunks.
Out: {split}_feat2_{j}.parquet (+ label y for train). Competition features across buckets come later (s5)."""
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from common import FEAT, WORK, done, timer
from s2_block import BUCKETS, cand_path

SUB = 16 # sub-chunks per bucket
STR = [  # (feature, column, scorer)
    ("core_tset", "core", fuzz.token_set_ratio),
    ("core_tsort", "core", fuzz.token_sort_ratio),
    ("core_ratio", "core", fuzz.ratio),
    ("core_partial", "core", fuzz.partial_ratio),
    ("name_ratio", "name_n", fuzz.ratio),
    ("skel_tset", "skel", fuzz.token_set_ratio),
    ("skel_ratio", "skel", fuzz.ratio),
    ("nosp_ratio", "nosp", fuzz.ratio),
    ("nosp_partial", "nosp", fuzz.partial_ratio),
    ("addr_tset", "addr_n", fuzz.token_set_ratio),
    ("addr_tsort", "addr_n", fuzz.token_sort_ratio),
    ("addr_partial", "addr_n", fuzz.partial_ratio),
    ("addr_ratio", "addr_n", fuzz.ratio),
]
COLS_N = ["name_n", "core", "skel", "nosp", "addr_n", "nums", "ch", "ah", "nonascii", "core_n1", "core_n23"]


def feat_path(split, j):
    return WORK / f"{split}_{FEAT}_{j}.parquet"


def norm3_path(split):
    return WORK / f"{split}_norm3.parquet"


def idf_path(split, col):
    return WORK / f"{split}_idf_{col}.parquet"


def prep(split):
    """Hashed token lists, IDF tables (counted in 8 id-chunks), nonascii flag, name frequencies."""
    if done(norm3_path(split)) and done(idf_path(split, "ah")):
        return
    with timer(f"prep {split}"):
        lf = pl.scan_parquet(WORK / f"{split}_norm2.parquet").with_columns(
            pl.col("core").str.split(" ").list.eval(pl.element().filter(pl.element() != "").hash())
            .list.unique().alias("ch"),
            pl.col("addr_n").str.split(" ").list.eval(pl.element().filter(pl.element() != "").hash())
            .list.unique().alias("ah"))
        cnt = lf.group_by("country").len("N").collect()
        for col in ("ch", "ah"):
            parts = [lf.filter(pl.col("id") % 8 == c).select("country", pl.col(col).alias("t")).explode("t")
                     .drop_nulls("t").group_by("country", "t").len("df").collect() for c in range(8)]
            (pl.concat(parts).group_by("country", "t").agg(pl.col("df").sum()).join(cnt, on="country")
             .select("country", "t", (pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias("w"))
             .write_parquet(idf_path(split, col)))
            del parts
        n = lf.select("id", "src", "country", "name_n", "core", "skel", "nosp", "addr_n", "nums", "ch", "ah").collect()
        raw = pl.scan_parquet(WORK / f"{split}_records.parquet").select(
            pl.col("id").alias("rid"), pl.col("name").str.contains(r"[^\x00-\x7f]").alias("nonascii")).collect()
        assert (raw["rid"] == n["id"]).all(), "records / norm2 row order differs"
        n = n.with_columns(raw["nonascii"])   # same row order: no join copy
        del raw
        n = n.with_columns((pl.col("src") == 1).sum().over(["country", "core"]).cast(pl.Int32).alias("core_n1"),
                           (pl.col("src") != 1).sum().over(["country", "core"]).cast(pl.Int32).alias("core_n23"))
        n.write_parquet(norm3_path(split))


def wjacc(x, col, idf, name):
    """IDF-weighted Jaccard of hashed token lists col (S1) vs col_2, plus max IDF of a shared token."""
    r = pl.int_range(pl.len(), dtype=pl.UInt32).alias("r")
    xx = x.select(r, "country", col, f"{col}_2")
    L = xx.select("r", "country", pl.col(col).alias("t")).explode("t").drop_nulls("t").join(idf, on=["country", "t"])
    R = xx.select("r", "country", pl.col(f"{col}_2").alias("t")).explode("t").drop_nulls("t").join(idf, on=["country", "t"])
    sl = L.group_by("r").agg(pl.col("w").sum().alias("sl"))
    sr = R.group_by("r").agg(pl.col("w").sum().alias("sr"))
    sh = L.join(R.select("r", "t"), on=["r", "t"]).group_by("r").agg(pl.col("w").sum().alias("sh"),
                                                                    pl.col("w").max().alias("mx"))
    o = (xx.select("r").join(sl, on="r", how="left").join(sr, on="r", how="left").join(sh, on="r", how="left")
         .sort("r").fill_null(0))
    u = (o["sl"] + o["sr"] - o["sh"]).to_numpy()
    return {f"{name}_wj": (o["sh"].to_numpy() / np.maximum(u, 1e-6)).astype(np.float32),
            f"{name}_wsh": o["sh"].to_numpy().astype(np.float32),
            f"{name}_wmax": o["mx"].to_numpy().astype(np.float32),
            f"{name}_wcov1": (o["sh"].to_numpy() / np.maximum(o["sl"].to_numpy(), 1e-6)).astype(np.float32),
            f"{name}_wcov2": (o["sh"].to_numpy() / np.maximum(o["sr"].to_numpy(), 1e-6)).astype(np.float32)}


def first_num_dist(a, b):
    """Levenshtein distance between first house numbers (typo'd digits), 9 if either missing."""
    return np.fromiter((Levenshtein.distance(p, q) if p is not None and q is not None else 9 for p, q in zip(a, b)),
                       dtype=np.int8, count=len(a))


def features_chunk(c, n, idf):
    x = (c.join(n.select(pl.col("id").alias("s1"), "country", *COLS_N), on="s1")
         .join(n.select("id", "src", *COLS_N), on="id", suffix="_2"))
    feats = {}
    for f, col, scorer in STR:
        feats[f] = cpdist(x[col].to_list(), x[col + "_2"].to_list(), scorer=scorer, workers=-1, dtype=np.uint8)
    feats.update(wjacc(x, "ch", idf["ch"], "nm"))
    feats.update(wjacc(x, "ah", idf["ah"], "ad"))
    feats["num1_lev"] = first_num_dist(x["nums"].list.first().to_list(), x["nums_2"].list.first().to_list())
    num_i = x["nums"].list.set_intersection(x["nums_2"]).list.len()
    num_u = x["nums"].list.set_union(x["nums_2"]).list.len()
    return x.select(
        "s1", "id", pl.col("src").cast(pl.Int8), *[f"nk{t}" for t in range(9)], "sc", "rk", "ncand_raw",
        pl.col("core").str.len_chars().cast(pl.Int16).alias("len_core"),
        pl.col("core_2").str.len_chars().cast(pl.Int16).alias("len_core2"),
        pl.col("addr_n").str.len_chars().cast(pl.Int16).alias("len_addr1"),
        pl.col("addr_n_2").str.len_chars().cast(pl.Int16).alias("len_addr2"),
        pl.col("nums").list.len().cast(pl.Int8).alias("n_nums1"),
        pl.col("nums_2").list.len().cast(pl.Int8).alias("n_nums2"),
        pl.col("nonascii_2").alias("nonascii2"),
        (pl.col("core") == pl.col("core_2")).alias("core_eq"),
        (pl.col("skel") == pl.col("skel_2")).alias("skel_eq"),
        "core_n1", "core_n23", pl.col("core_n1_2").alias("core2_n1"),
        (pl.col("nums").list.first() == pl.col("nums_2").list.first()).fill_null(False).alias("num_first_eq"),
        pl.col("nums_2").list.contains(pl.col("nums").list.first()).fill_null(False).alias("num1_in_2"),
    ).with_columns(
        num_i.cast(pl.Int8).alias("num_inter"),
        (num_i / num_u.clip(lower_bound=1)).cast(pl.Float32).alias("num_jacc"),
        (num_u - num_i).cast(pl.Int8).alias("num_diff"),
        *[pl.Series(k, v) for k, v in feats.items()],
    )


def rss():
    try:
        import resource
        return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak"
    except ImportError:   # Windows (local smoke tests)
        return "rss n/a"


def load_records(split, ids):
    """Rows of norm3 whose id is in ids. pyarrow's dataset scanner filters batch by batch, so only the
    result (~1.5 GB) is materialised (a full read + filter peaked near 10 GB, polars batches near 5.4 GB)."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    t = ds.dataset(str(norm3_path(split)), format="parquet").to_table(
        filter=ds.field("id").isin(pa.array(ids["id"].to_numpy())), batch_size=200_000)
    return pl.from_arrow(t).with_columns(pl.col("country").cast(pl.Utf8))


def finish_chunk(out, gt):
    """Per-S1 relative features (chunks partition S1, so windows are exact) + label."""
    out = out.with_columns((pl.col("core_tset").cast(pl.Int16) + pl.col("addr_tset")).alias("pair_sum"),
                           (pl.col("skel_tset").cast(pl.Int16) + pl.col("addr_tset")).alias("pair_sum2"))
    out = out.with_columns(
        (pl.col("pair_sum").max().over("s1") - pl.col("pair_sum")).alias("gap_s1_best"),
        pl.col("pair_sum").rank("min", descending=True).over("s1").cast(pl.Int16).alias("rk_s1_ps"),
        (pl.col("pair_sum2").max().over("s1") - pl.col("pair_sum2")).alias("gap_s1_best2"),
    )
    if gt is not None:
        out = out.join(gt, on=["s1", "id"], how="left").with_columns(pl.col("y").fill_null(0))
    return out


def features(split, j):
    """Writes feat_path(split, j). Each sub-chunk is finished and spilled to disk, then stream-concatenated."""
    c = pl.read_parquet(cand_path(split, j))
    n = load_records(split, pl.concat([c["s1"], c["id"]]).unique().to_frame("id"))
    idf = {col: pl.read_parquet(idf_path(split, col)).with_columns(pl.col("country").cast(pl.Utf8))
           for col in ("ch", "ah")}
    print(f"  loaded {n.height:,} records for {c.height:,} pairs, {rss()}", flush=True)
    gt = None
    if split == "train":
        gt = (pl.scan_parquet(WORK / "train_gt.parquet").filter(pl.col("s1") % BUCKETS == j)
              .with_columns(pl.lit(1, pl.Int8).alias("y")).collect())
    parts = []
    for s in range(SUB):
        cs = c.filter((pl.col("s1") // BUCKETS) % SUB == s)
        pf = WORK / f"_tmp_{split}_{j}_{s}.parquet"
        finish_chunk(features_chunk(cs, n, idf), gt).write_parquet(pf)
        parts.append(pf)
        if s % 4 == 3:
            print(f"  chunk {s + 1}/{SUB} {rss()}", flush=True)
        del cs
    del n, c
    pl.scan_parquet(parts).sink_parquet(feat_path(split, j))
    for pf in parts:
        pf.unlink()


if __name__ == "__main__":
    # usage: s3_features.py <split> prep | s3_features.py <split> <bucket>
    split, what = sys.argv[1], sys.argv[2]
    if what == "prep":
        prep(split)
    else:
        j = int(what)
        if not done(feat_path(split, j)):
            with timer(f"features {split} {j}"):
                features(split, j)
