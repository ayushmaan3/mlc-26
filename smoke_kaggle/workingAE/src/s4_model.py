"""Stage 4: LightGBM pair classifier + assignment decode + threshold tuning + write submission.
Train on train buckets TRAIN_B, validate on VAL_B (macro F0.5), then predict test."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import OUT, WORK, decode_ids, done, f05, timer
from s2_block import BUCKETS
from s3_features import feat_path

TRAIN_B, VAL_B = [0], [6, 7]
DROP = {"s1", "id", "y"}
MODEL = WORK / "lgb_v1.txt"


def load(split, buckets):
    return pl.concat([pl.read_parquet(feat_path(split, j)) for j in buckets])


def feats(df):
    return [c for c in df.columns if c not in DROP]


def assign(pred, t):
    """Each S2/S3 id -> its single best S1 (GT shows ids match at most one S1), keep if p >= t."""
    return (pred.filter(pl.col("p") >= t).sort("p", descending=True)
            .unique("id", keep="first").select("s1", "id"))


def train():
    if done(MODEL):
        return lgb.Booster(model_file=str(MODEL))
    with timer("train lgb"):
        tr = load("train", TRAIN_B)
        cols = feats(tr)
        print("train rows", tr.height, "pos", tr["y"].sum(), "features", len(cols))
        ds = lgb.Dataset(tr.select(cols).to_numpy().astype(np.float32), tr["y"].to_numpy(), feature_name=cols,
                         free_raw_data=True)
        del tr
        params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                      feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=0, verbose=-1)
        m = lgb.train(params, ds, num_boost_round=300)
        m.save_model(str(MODEL))
    return m


def predict(m, df):
    X = df.select(m.feature_name()).to_numpy().astype(np.float32)
    return df.select("s1", "id").with_columns(pl.Series("p", m.predict(X, num_threads=0)).cast(pl.Float32))


def validate(m):
    va = load("train", VAL_B)
    pred = predict(m, va)
    truth = (pl.read_parquet(WORK / "train_gt.parquet").filter((pl.col("s1") % BUCKETS).is_in(VAL_B)))
    s1_all = (pl.scan_parquet(WORK / "train_norm.parquet").filter(pl.col("src") == 1)
              .filter((pl.col("id") % BUCKETS).is_in(VAL_B)).select(pl.col("id").alias("s1")).collect())
    res = {}
    for t in np.arange(0.2, 0.95, 0.05):
        res[round(float(t), 2)] = f05(s1_all, assign(pred, t), truth)
    best = max(res, key=res.get)
    print("val F0.5 by threshold:", {k: round(v, 4) for k, v in res.items()})
    print(f"BEST t={best} F0.5={res[best]:.4f}")
    imp = sorted(zip(m.feature_name(), m.feature_importance("gain")), key=lambda x: -x[1])[:15]
    print("top gain:", [(f, int(g)) for f, g in imp])
    json.dump({"t": best, "val_f05": res[best], "curve": res}, open(WORK / "val_v1.json", "w"))
    return best


def write_test(m, t):
    OUT.mkdir(parents=True, exist_ok=True)
    preds, cands = [], []
    for j in range(BUCKETS):
        with timer(f"predict test {j}"):
            df = pl.read_parquet(feat_path("test", j))
            p = predict(m, df)
            preds.append(p.filter(pl.col("p") >= t))
            cands.append(p.select("s1", "id"))
            del df
    match = assign(pl.concat(preds), t)
    s1_all = (pl.scan_parquet(WORK / "test_norm.parquet").filter(pl.col("src") == 1)
              .select(pl.col("id").alias("s1")).collect())

    def to_tsv(pairs, col, path):
        g = pairs.sort("s1", "id").group_by("s1", maintain_order=True).agg(
            decode_ids("id").str.join(",").alias(col))
        (s1_all.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
         .select(decode_ids("s1").alias("source1_entity_id"), col)
         .write_csv(path, separator="\t", quote_style="never"))

    to_tsv(match, "matched_entity_ids", OUT / "matching_results.tsv")
    to_tsv(pl.concat(cands), "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    print("matched pairs", match.height, "S1 with >=1 match", match["s1"].n_unique(), "of", s1_all.height)


if __name__ == "__main__":
    m = train()
    t = validate(m)
    if "--test" in sys.argv:
        write_test(m, t)
