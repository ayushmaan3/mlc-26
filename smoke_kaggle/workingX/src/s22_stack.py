"""Set-level second stage on saved probabilities (last sprint). The pair model scores each (S1, record) alone; the error budget
shows the remaining loss is set-level (S1 missing one match, a record taken by the wrong same-name S1, another business at the same
address). Features use ONLY one model's p (>= 0.5) plus record hashes, so they are built identically on train OOF and on test:
  S1 side   : #candidates, #confident (p >= 0.9) overall / per source, rank of p, max / sum of p, #S1 sharing the name
  record    : #S1 candidates, rank, gap to its best S1, the best rival S1's p and #confident, record without address
  siblings  : the S1's other confident matches with the same exact address / name as this record; exact name / address = S1's
Train: run R's OOF p7 (state/oof_p7_ge05.parquet, single-model OOF). Test: run D's p7 per model (p_m0, p_m1) -> stacked separately
(each is a single-model p like the OOF) -> mean. Decode t 0.80 / France 0.90 (the LB-proven setting).
  selftest | run <name>"""
import sys

import numpy as np
import polars as pl

from common import OUT, WORK, f05, timer
from s2_block import BUCKETS

T, T_FR, CONF, LOWP = 0.80, 0.90, 0.9, 0.5
F = ["p", "s1_n", "s1_nconf", "s1_rk", "s1_pmax", "s1_psum", "r_n", "r_rk", "r_gap", "addr_eq_s1", "name_eq_s1", "s1_conf_s2",
     "s1_conf_s3", "sib_same_addr", "sib_same_name", "rival_nconf", "rival_p", "s1_name_n", "r_noaddr", "src"]
import os
USE_CE = os.environ.get("BER_STACK_CE") == "1"   # run X: + fold-wise cross-encoder logit on the uncertain band (s23, ceband_*)
if USE_CE:
    F = F + ["ce"]
PARAMS = dict(objective="binary:logistic", eta=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              eval_metric="logloss", nthread=4, seed=0, device=os.environ.get("BER_STACK_DEVICE", "cpu"))
ROUNDS = 300
_NORM = lambda c: pl.col(c).fill_null("").str.to_lowercase().str.strip_chars()


def context(split):
    """S1 [s1, country, n1, a1, s1_name_n] and records [id, n2, a2, src, r_noaddr] from {split}_records.parquet."""
    r = pl.read_parquet(WORK / f"{split}_records.parquet", columns=["id", "src", "name", "addr", "country"])
    r = r.with_columns(_NORM("name").hash().alias("hn"), _NORM("addr").hash().alias("ha"), (_NORM("addr") == "").alias("noaddr"),
                       pl.col("country").cast(pl.Utf8))
    s1 = (r.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "country", pl.col("hn").alias("n1"), pl.col("ha").alias("a1"))
          .with_columns(pl.len().over(["country", "n1"]).alias("s1_name_n")))
    rec = r.filter(pl.col("src") != 1).select("id", pl.col("hn").alias("n2"), pl.col("ha").alias("a2"), pl.col("src").cast(pl.Int8),
                                              pl.col("noaddr").alias("r_noaddr"))
    return s1, rec


def features(pr, s1, rec):
    """pr [s1, id, p]: ONE model's probabilities, p >= LOWP."""
    x = pr.join(s1, on="s1").join(rec, on="id")
    conf = pl.col("p") >= CONF
    x = x.with_columns(
        pl.len().over("s1").alias("s1_n"), conf.sum().over("s1").alias("s1_nconf"),
        pl.col("p").rank("ordinal", descending=True).over("s1").alias("s1_rk"),
        pl.col("p").max().over("s1").alias("s1_pmax"), pl.col("p").sum().over("s1").alias("s1_psum"),
        pl.len().over("id").alias("r_n"), pl.col("p").rank("ordinal", descending=True).over("id").alias("r_rk"),
        (pl.col("p").max().over("id") - pl.col("p")).alias("r_gap"),
        (pl.col("a2") == pl.col("a1")).cast(pl.Int8).alias("addr_eq_s1"), (pl.col("n2") == pl.col("n1")).cast(pl.Int8).alias("name_eq_s1"),
        (conf & (pl.col("src") == 2)).sum().over("s1").alias("s1_conf_s2"), (conf & (pl.col("src") == 3)).sum().over("s1").alias("s1_conf_s3"))
    sa = x.filter(conf).group_by("s1", "a2").len("sib_same_addr")
    sn = x.filter(conf).group_by("s1", "n2").len("sib_same_name")
    own = conf.cast(pl.Int64)
    x = (x.join(sa, on=["s1", "a2"], how="left").join(sn, on=["s1", "n2"], how="left")
         .with_columns((pl.col("sib_same_addr").fill_null(0).cast(pl.Int64) - own).alias("sib_same_addr"),
                       (pl.col("sib_same_name").fill_null(0).cast(pl.Int64) - own).alias("sib_same_name")))
    # the record's best rival S1 (the best S1 that is not this one): top-2 per record via first / last (no list.get)
    top = (x.select("id", "s1", "p", "s1_nconf").sort(["id", "p"], descending=[False, True]).group_by("id", maintain_order=True).head(2)
           .group_by("id").agg(pl.len().alias("k"), pl.col("s1").first().alias("t1"), pl.col("p").first().alias("tp1"),
                               pl.col("s1_nconf").first().alias("tn1"), pl.col("p").last().alias("tp2"), pl.col("s1_nconf").last().alias("tn2")))
    x = x.join(top, on="id").with_columns(
        pl.when(pl.col("t1") != pl.col("s1")).then(pl.col("tp1")).when(pl.col("k") == 2).then(pl.col("tp2")).otherwise(None).alias("rival_p"),
        pl.when(pl.col("t1") != pl.col("s1")).then(pl.col("tn1")).when(pl.col("k") == 2).then(pl.col("tn2")).otherwise(None).alias("rival_nconf"))
    return x.drop("k", "t1", "tp1", "tn1", "tp2", "tn2")


def ce_all(split):
    return pl.concat([pl.read_parquet(WORK / f"ceband_{split}_{j}.parquet", columns=["s1", "id", "ce"]) for j in range(BUCKETS)])


def matrix(x):
    return x.select(pl.col(F).cast(pl.Float32)).to_numpy()


def decode(x, col, t=T, t_fr=T_FR):
    q = x.filter(pl.when(pl.col("country") == "France").then(pl.col(col) >= t_fr).otherwise(pl.col(col) >= t))
    return q.sort(col, descending=True).unique("id", keep="first").select("s1", "id")


def selftest():
    s1 = pl.DataFrame({"s1": [1, 2], "country": ["US", "US"], "n1": [7, 7], "a1": [100, 200], "s1_name_n": [2, 2]})
    rec = pl.DataFrame({"id": [10, 11, 12], "n2": [7, 7, 7], "a2": [100, 100, 5], "src": [2, 3, 3], "r_noaddr": [False, False, True]}).with_columns(pl.col("src").cast(pl.Int8))
    pr = pl.DataFrame({"s1": [1, 1, 1, 2], "id": [10, 11, 12, 12], "p": [0.97, 0.95, 0.6, 0.7]})
    x = features(pr, s1, rec).sort("s1", "id")
    r = {(a, b): d for a, b, d in x.select("s1", "id", pl.struct(F).alias("d")).iter_rows()}
    assert r[(1, 10)]["sib_same_addr"] == 1 and r[(1, 11)]["sib_same_addr"] == 1 and r[(1, 12)]["sib_same_addr"] == 0, r
    assert r[(1, 12)]["rival_p"] == 0.7 and r[(2, 12)]["rival_p"] == 0.6 and r[(1, 10)]["rival_p"] is None, r
    assert r[(1, 12)]["r_n"] == 2 and r[(1, 10)]["s1_nconf"] == 2 and r[(2, 12)]["s1_nconf"] == 0 and r[(1, 12)]["r_noaddr"] == 1, r
    assert matrix(x).shape == (4, len(F)) and x["rival_nconf"].to_list() == [None, None, 0, 2], x
    print(f"SELFTEST OK (polars {pl.__version__}): stack features / rival / siblings", flush=True)


def run(name):
    import xgboost as xgb
    with timer("stack: train on run R's OOF"):
        s1, rec = context("train")
        truth = pl.read_parquet(WORK / "train_gt.parquet")
        oof = pl.read_parquet(WORK / "oof_p7_ge05.parquet").select("s1", "id", "p")
        x = features(oof, s1, rec).join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "id"], how="left").with_columns(pl.col("y").fill_null(0))
        if USE_CE:
            x = x.join(ce_all("train"), on=["s1", "id"], how="left")
            print(f"  CE scored {x['ce'].is_not_null().sum():,} of {x.height:,} OOF pairs", flush=True)
        f0 = (pl.col("s1") % BUCKETS).is_in([0, 2, 4, 6])
        parts = []
        for f in (0, 1):
            tr, va = x.filter(f0 if f == 0 else ~f0), x.filter(~f0 if f == 0 else f0)
            b = xgb.train(PARAMS, xgb.DMatrix(matrix(tr), label=tr["y"].to_numpy(), feature_names=F), ROUNDS)
            parts.append(va.select("s1", "id", "p", "country").with_columns(pl.Series("p2", b.predict(xgb.DMatrix(matrix(va), feature_names=F)))))
        o = pl.concat(parts)
        s1_all = s1.select("s1", "country")
        base, st = f05(s1_all, decode(o, "p"), truth), f05(s1_all, decode(o, "p2"), truth)
        print(f"OOF (US+India) at t {T}: base p {base:.5f} | stacked p2 {st:.5f} | delta {st - base:+.5f}  (laptop run: +0.00078)", flush=True)
        for c in ("US", "India"):
            sc = s1_all.filter(pl.col("country") == c)
            tc = truth.join(sc, on="s1", how="semi")
            print(f"  {c}: base {f05(sc, decode(o, 'p').join(sc, on='s1', how='semi'), tc):.5f} | "
                  f"stacked {f05(sc, decode(o, 'p2').join(sc, on='s1', how='semi'), tc):.5f}", flush=True)
        import os
        assert st > base or os.environ.get("BER_STACK_FORCE") == "1", "stacking is not better on OOF -> not writing a submission"
        full = xgb.train(PARAMS, xgb.DMatrix(matrix(x), label=x["y"].to_numpy(), feature_names=F), ROUNDS)
        del x, parts, o
    with timer("stack: test (run D's p7, per model)"):
        s1t, rect = context("test")
        tp = pl.concat([pl.read_parquet(WORK / f"p7_v3_test_{j}.parquet", columns=["s1", "id", "p_m0", "p_m1"]) for j in range(BUCKETS)])
        outs = []
        for m in ("p_m0", "p_m1"):
            xm = features(tp.filter(pl.col(m) >= LOWP).select("s1", "id", pl.col(m).alias("p")), s1t, rect)
            if USE_CE:
                xm = (xm.join(ce_all("test"), on=["s1", "id"], how="left")
                      .with_columns(pl.when(pl.col("p") < 0.995).then(pl.col("ce")).otherwise(None).alias("ce")))
            outs.append(xm.select("s1", "id", pl.Series(f"s_{m}", full.predict(xgb.DMatrix(matrix(xm), feature_names=F)))))
        # a pair below LOWP for one model gets that model's raw p (it can't pass t anyway): mean over both models
        y = (tp.join(outs[0], on=["s1", "id"], how="left").join(outs[1], on=["s1", "id"], how="left")
             .with_columns(((pl.col("s_p_m0").fill_null(pl.col("p_m0")) + pl.col("s_p_m1").fill_null(pl.col("p_m1"))) / 2).alias("p2"),
                           ((pl.col("p_m0") + pl.col("p_m1")) / 2).alias("p"))
             .join(s1t.select("s1", "country"), on="s1"))
        m2, m1 = decode(y, "p2"), decode(y, "p")
        n1 = s1t.group_by("country").len("n_s1")
        print(m2.join(s1t.select("s1", "country"), on="s1").group_by("country").len("stack")
              .join(m1.join(s1t.select("s1", "country"), on="s1").group_by("country").len("p7_plain"), on="country").join(n1, on="country")
              .with_columns((pl.col("stack") / pl.col("n_s1")).round(3).alias("stack_per_S1"),
                            (pl.col("p7_plain") / pl.col("n_s1")).round(3).alias("plain_per_S1")).sort("country"), flush=True)
        from s6_write import to_tsv
        s1_all = (pl.scan_parquet(WORK / "test_norm2.parquet").filter(pl.col("src") == 1).select(pl.col("id").alias("s1")).sort("s1").collect())
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        to_tsv(s1_all, m2, "matched_entity_ids", d / "matching_results.tsv")
        print(f"  wrote {d}/matching_results.tsv  matched pairs {m2.height:,} (plain p7 at the same decode: {m1.height:,})", flush=True)


if __name__ == "__main__":
    selftest() if sys.argv[1] == "selftest" else run(sys.argv[2])
