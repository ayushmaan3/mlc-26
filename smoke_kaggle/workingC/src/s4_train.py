"""Stage 4: XGBoost (GPU) pair classifier with 2-fold out-of-fold predictions by S1 bucket.
stage 1: pair features only. stage 2: + competition features built from stage-1 OOF probabilities (s5_comp).
stage 3: stage 2 + cross-encoder logit on uncertain pairs (s8_ce; NaN = not scored) + optional embedding cosine (e2).
stage 5: same as stage 4 once the s14 address add-on (addr4_*) exists -> stage 4 + house-number-noise features.
stage 4: stage 1 features + cross-encoder logit, NO competition features (stacking on p1 proved fragile on test: v2
         stage 2 OOF +0.003 but LB -0.004).
Test-time consistency: a stacked model trained on fold f's buckets saw inputs produced by the *other* fold's model, so on
test each booster g gets the comp / ce inputs of model 1-g (variant files/columns "_m0"/"_m1"), not the 2-model mean.
Data is streamed half-bucket by half-bucket into a GPU QuantileDMatrix, so host RAM holds one piece at a time.
Out: WORK/p{stage}_{split}_{j}.parquet [s1, id, p]   (train = OOF, test = mean of the two fold models)."""
import json
import sys

import numpy as np
import polars as pl
import xgboost as xgb

from common import TAG, WORK, done, f05, timer
from s2_block import BUCKETS
from s3_features import feat_path

FOLDS = [[0, 2, 4, 6], [1, 3, 5, 7]]
DROP = {"s1", "id", "y"}
_E = __import__("os").environ.get
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
              max_bin=256, eta=float(_E("BER_ETA", 0.08)), grow_policy="lossguide", max_leaves=int(_E("BER_LEAVES", 255)),
              max_depth=0, min_child_weight=int(_E("BER_MCW", 20)), subsample=0.8,
              colsample_bytree=float(_E("BER_COLS", 0.8)), reg_lambda=1.0, seed=int(_E("BER_SEED", 0)))
ROUNDS, EARLY = 3000, 100
NPIECE = int(_E("BER_NPIECE", 4))   # each bucket is streamed in pieces: host RAM holds one piece at a time
EASY_P, EASY_KEEP = 0.01, int(_E("BER_EASY_KEEP", 10))   # training keeps hard pairs + 1/EASY_KEEP of easy ones (weight EASY_KEEP)
TRAIN_BUCKETS = int(_E("BER_TRAIN_BUCKETS", 4))   # buckets per fold used for training (2 = half data, for 12 GB machines)


def comp_path(split, j, v=None):
    """v = which stage-1 model produced the test p1 the comp features were built from (test only)."""
    return WORK / (f"comp{TAG}_{split}_{j}.parquet" if v is None else f"comp{TAG}_{split}_m{v}_{j}.parquet")


def uses_comp(stage):
    return stage in (2, 3)


def ce_path(split, j):
    return WORK / f"ce{TAG}_{split}_{j}.parquet"


def embcos_path(split, j):
    return WORK / f"embcos{TAG}_{split}_{j}.parquet"


def idc_path(split, j):
    """s16 per-record competition features (similarity-based, untagged)."""
    return WORK / f"idc_{split}_{j}.parquet"


def addr_path(split, j):
    """s14 address add-on (model-independent, untagged)."""
    return WORK / f"addr4_{split}_{j}.parquet"


def extra_paths(split, j, stage):
    """Optional stage-3 feature files (joined when present for train bucket 0, i.e. produced for all buckets)."""
    if stage < 3:   # stage 3 = comp + extras, stage 4 = extras only
        return []
    return [f(split, j) for f in (ce_path, embcos_path, addr_path, idc_path) if f("train", 0).exists()]


def p_path(stage, split, j):
    return WORK / f"p{stage}{TAG}_{split}_{j}.parquet"


def model_path(stage, f):
    return WORK / f"xgb_s{stage}{TAG}_f{f}.json"


def frame(split, j, stage, piece=None, sample=False, variant=None):
    """Features of bucket j (optionally one quarter), + comp (stages 2-3) / extra features (stages 3-4).
    variant (test only): use the comp file / ce_m{variant} column made from model `variant`.
    sample=True (training only): drop most easy negatives (p1 < EASY_P); returns (df, weights)."""
    def part(path):
        lf = pl.scan_parquet(path)
        return lf if piece is None else lf.filter((pl.col("s1") // BUCKETS) % NPIECE == piece)
    lf = part(feat_path(split, j))
    if uses_comp(stage):
        lf = lf.join(part(comp_path(split, j, variant)), on=["s1", "id"], how="left")
    for ep in extra_paths(split, j, stage):
        e = part(ep)
        names = e.collect_schema().names()
        vcols = [c for c in names if c[-3:] in ("_m0", "_m1")]
        if variant is not None:   # swap in the per-model column, e.g. ce <- ce_m1
            e = e.with_columns([pl.col(f"{c}_m{variant}").alias(c) for c in names if f"{c}_m{variant}" in names])
        lf = lf.join(e.drop(vcols), on=["s1", "id"], how="left")
    if not sample:
        return lf.collect()
    if not uses_comp(stage):
        # no p1 yet: easy negatives = outside the S1's top-10 by fuzzy pair score AND by embedding cosine (true matches
        # are nearly always inside); keep all others + 1/EASY_KEEP of the easy ones (weight EASY_KEEP). v3 has 139M pairs.
        names = lf.collect_schema().names()
        easy = pl.col("rk_s1_ps") > 10
        if "emb_rk" in names:
            easy = easy & (pl.col("emb_rk") > 10)
        df = lf.with_columns(easy.alias("_easy")).filter(
            ~pl.col("_easy") | (pl.struct("s1", "id").hash(seed=7) % EASY_KEEP == 0)).collect()
        w = np.where(df["_easy"].to_numpy(), float(EASY_KEEP), 1.0).astype(np.float32)
        return df.drop("_easy"), w
    easy = pl.col("p1") < EASY_P
    df = lf.filter(~easy | (pl.struct("s1", "id").hash(seed=7) % EASY_KEEP == 0)).collect()
    return df, np.where(df["p1"].to_numpy() < EASY_P, float(EASY_KEEP), 1.0).astype(np.float32)


def feature_cols(stage):
    cols = pl.scan_parquet(feat_path("train", 0)).collect_schema().names()
    if uses_comp(stage):
        cols += [c for c in pl.scan_parquet(comp_path("train", 0)).collect_schema().names() if c not in cols]
    for ep in extra_paths("train", 0, stage):
        cols += [c for c in pl.scan_parquet(ep).collect_schema().names() if c not in cols and c[-3:] not in ("_m0", "_m1")]
    return [c for c in cols if c not in DROP]


class Batches(xgb.DataIter):
    def __init__(self, buckets, stage, cols):
        self.pieces = [(j, h) for j in buckets for h in range(NPIECE)]
        self.stage, self.cols, self.i = stage, cols, 0
        super().__init__()

    def next(self, input_data):
        if self.i == len(self.pieces):
            return False
        j, h = self.pieces[self.i]
        df, w = frame("train", j, self.stage, h, sample=True)
        X = df.select(pl.col(self.cols).cast(pl.Float32)).to_numpy()   # direct float32: no float64 temp copy
        y = df["y"].to_numpy()
        del df
        input_data(data=X, label=y, weight=w, feature_names=self.cols)
        del X, y, w
        self.i += 1
        return True

    def reset(self):
        self.i = 0


def predict(bsts, split, j, stage, cols):
    """Probability, a quarter bucket at a time. One booster -> p. Two boosters (test) -> p = mean, plus p_m0 / p_m1
    per model; for stacked stages booster g reads the inputs made by model 1-g (see module doc)."""
    stacked = split == "test" and len(bsts) == 2 and (uses_comp(stage) or bool(extra_paths(split, j, stage)))
    out = []
    for h in range(NPIECE):
        if stacked:
            ps = []
            for g, b in enumerate(bsts):
                df = frame(split, j, stage, h, variant=1 - g)
                X = xgb.DMatrix(df.select(pl.col(cols).cast(pl.Float32)).to_numpy(), feature_names=cols)
                ps.append(b.predict(X))
                ids = df.select("s1", "id")
                del df, X
        else:
            df = frame(split, j, stage, h)
            X = xgb.DMatrix(df.select(pl.col(cols).cast(pl.Float32)).to_numpy(), feature_names=cols)
            ps = [b.predict(X) for b in bsts]
            ids = df.select("s1", "id")
            del df, X
        o = ids.with_columns(pl.Series("p", np.mean(ps, axis=0)).cast(pl.Float32))
        if len(ps) == 2:
            o = o.with_columns(pl.Series("p_m0", ps[0]).cast(pl.Float32), pl.Series("p_m1", ps[1]).cast(pl.Float32))
        out.append(o)
    return pl.concat(out)


def train(stage):
    cols = feature_cols(stage)
    print(f"stage {stage}: {len(cols)} features", flush=True)
    for f, tr_b in enumerate(FOLDS):
        mp = model_path(stage, f)
        if done(mp):
            continue
        with timer(f"train stage {stage} fold {f} on buckets {tr_b}"):
            dtr = xgb.QuantileDMatrix(Batches(tr_b[:TRAIN_BUCKETS], stage, cols), max_bin=PARAMS["max_bin"])
            ev, ew = frame("train", FOLDS[1 - f][0], stage, 0, sample=True)   # quarter bucket, other fold
            dev = xgb.QuantileDMatrix(ev.select(pl.col(cols).cast(pl.Float32)).to_numpy(), label=ev["y"].to_numpy(),
                                      weight=ew, ref=dtr, feature_names=cols)
            del ev
            bst = xgb.train(PARAMS, dtr, ROUNDS, evals=[(dev, "ev")], early_stopping_rounds=EARLY,
                            verbose_eval=100)
            print("best iteration", bst.best_iteration, flush=True)
            bst.save_model(str(mp))
            imp = sorted(bst.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:20]
            print("top gain:", [(k, int(v)) for k, v in imp], flush=True)
            del dtr, dev, bst
    return cols


def predict_all(stage, cols):
    bsts = [xgb.Booster(model_file=str(model_path(stage, f))) for f in range(2)]
    for b in bsts:
        b.set_param({"device": "cuda"})
    for f, va_b in enumerate(FOLDS):          # OOF: model trained on the other fold
        for j in va_b:
            if not done(p_path(stage, "train", j)):
                with timer(f"oof stage {stage} bucket {j}"):
                    predict([bsts[1 - f]], "train", j, stage, cols).write_parquet(p_path(stage, "train", j))
    for j in range(BUCKETS):
        if not done(p_path(stage, "test", j)):
            with timer(f"test stage {stage} bucket {j}"):
                predict(bsts, "test", j, stage, cols).write_parquet(p_path(stage, "test", j))


def assign(pred, t):
    """Each S2/S3 id -> its single best S1 (GT: every id matches at most one S1); keep if p >= t."""
    return (pred.filter(pl.col("p") >= t).sort("p", descending=True)
            .unique("id", keep="first").select("s1", "id"))


def evaluate(stage):
    pred = pl.concat([pl.read_parquet(p_path(stage, "train", j)).filter(pl.col("p") >= 0.15) for j in range(BUCKETS)])
    truth = pl.read_parquet(WORK / "train_gt.parquet")
    s1 = (pl.scan_parquet(WORK / "train_norm2.parquet").filter(pl.col("src") == 1)
          .select(pl.col("id").alias("s1"), "country").collect())
    res = {}
    for t in np.arange(0.2, 0.96, 0.025):
        res[round(float(t), 3)] = f05(s1, assign(pred, t), truth)
    best = max(res, key=res.get)
    print(f"stage {stage} OOF macro F0.5 (all train S1, global assign): best t={best} F={res[best]:.5f}")
    print({k: round(v, 4) for k, v in res.items()})
    by = {}
    a = assign(pred, best)
    for c in s1["country"].unique().to_list():
        sc = s1.filter(pl.col("country") == c)
        by[c] = f05(sc, a.join(sc, on="s1", how="semi"), truth.join(sc, on="s1", how="semi"))
    print("by country", by)
    json.dump({"t": best, "f05": res[best], "curve": res, "by_country": by},
              open(WORK / f"val_stage{stage}{TAG}.json", "w"))
    return best


if __name__ == "__main__":
    st = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    c = train(st)
    predict_all(st, c)
    evaluate(st)
