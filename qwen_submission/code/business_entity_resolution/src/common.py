"""Paths + shared helpers. Every stage reads/writes parquet under WORK so re-runs skip finished stages."""
import os
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BER_DATA", HERE.parents[2] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"))
WORK = Path(os.environ.get("BER_WORK", HERE.parents[2] / "work"))
OUT = Path(os.environ.get("BER_OUT", HERE.parents[2] / "output"))
WORK.mkdir(parents=True, exist_ok=True)
SPLITS = ("train", "test")


def encode_ids(col):
    """'S2-681193310' -> 2_681193310 (int64). Source digit * 1e9 + number."""
    c = pl.col(col)
    return (c.str.slice(1, 1).cast(pl.Int64) * 1_000_000_000 + c.str.slice(3).cast(pl.Int64))


def decode_ids(col):
    c = pl.col(col)
    return pl.concat_str([pl.lit("S"), (c // 1_000_000_000).cast(pl.Utf8), pl.lit("-"),
                          (c % 1_000_000_000).cast(pl.Utf8)])


def done(path):
    return Path(path).exists()


class timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t = time.time()
        print(f"> {self.msg}", flush=True)

    def __exit__(self, *a):
        print(f"  {self.msg}: {time.time() - self.t:.0f}s", flush=True)


def f05_scores(s1_all, pred, truth):
    """s1_all: [s1] every S1 scored (singletons count). pred/truth: [s1, id] pairs.
    Per S1: F0.5 = 1.25*tp / (0.25*n_true + n_pred); both empty -> 1."""
    tp =pred.join(truth, on=["s1", "id"], how="inner").group_by("s1").len("tp")
    npred = pred.group_by("s1").len("np")
    ntrue = truth.group_by("s1").len("nt")
    df = (s1_all.select("s1").join(tp, on="s1", how="left").join(npred, on="s1", how="left")
          .join(ntrue, on="s1", how="left").fill_null(0))
    f = pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    return df.with_columns(f.alias("f"))


def f05(s1_all, pred, truth):
    return f05_scores(s1_all, pred, truth)["f"].mean()
