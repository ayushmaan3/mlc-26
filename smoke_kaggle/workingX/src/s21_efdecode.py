"""Per-S1 expected-F0.5 decoding on saved predictions (Fable review 4), tuned on labelled OOF before any use.
After the global assignment (each S2/S3 record -> its best S1, p >= floor), each S1 keeps the top-k of its candidates that
maximise E[F0.5 | k] ~= 1.25 * sum_{i<=k} q_i / (0.25 * sum_i q_i + k), or nothing when P(no true match) = prod(1 - q_i) is higher
(singletons score 1 only when empty). q = p ** gamma (calibration). France may use its own (floor, gamma): its probabilities
proved over-confident on the LB (France 0.80 -> 0.90 threshold: France F +0.0013). Same maths as s9_decode.decode.
  selftest
  tune <oof.parquet>                          : grid of (floor, gamma) on OOF vs the global t 0.80 -> prints best + delta
  apply <name> <floor> <gamma> <floor_fr> <gamma_fr> <stage...> : test p (mean of stages) -> OUT/<name>/matching_results.tsv"""
import sys

import numpy as np
import polars as pl

from common import OUT, WORK, f05, timer
from s2_block import BUCKETS

T = 0.80
FLOORS = [0.5, 0.6, 0.7, 0.8]
GAMMAS = [1.0, 1.25, 1.5, 2.0]


def assigned(pred, floor, floor_fr=None):
    fr = pl.col("country") == "France" if "country" in pred.columns else pl.lit(False)
    q = pred.filter(pl.when(fr).then(pl.col("p") >= (floor_fr if floor_fr is not None else floor)).otherwise(pl.col("p") >= floor))
    return q.sort("p", descending=True).unique("id", keep="first")


def decode(pa, g, g_fr=None):
    fr = pl.col("country") == "France" if "country" in pa.columns else pl.lit(False)
    gam = pl.when(fr).then(pl.lit(g_fr if g_fr is not None else g)).otherwise(pl.lit(g))
    x = (pa.with_columns((pl.col("p").cast(pl.Float64).clip(1e-6, 1.0).log() * gam).exp().clip(1e-6, 1 - 1e-6).alias("q"))
         .sort(["s1", "q"], descending=[False, True])
         .with_columns(pl.col("q").cum_sum().over("s1").alias("cq"), pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                       pl.col("q").sum().over("s1").alias("sq"), (1 - pl.col("q")).log().sum().over("s1").exp().alias("p_empty")))
    x = x.with_columns((1.25 * pl.col("cq") / (0.25 * pl.col("sq") + pl.col("k"))).alias("ef"))
    best = (x.sort(["s1", "ef", "k"], descending=[False, True, False]).group_by("s1", maintain_order=True)
            .agg(pl.col("ef").first().alias("ef_best"), pl.col("k").first().alias("k_best"), pl.col("p_empty").first()))
    keep = best.filter(pl.col("ef_best") > pl.col("p_empty")).select("s1", "k_best")
    return x.join(keep, on="s1").filter(pl.col("k") <= pl.col("k_best")).select("s1", "id")


def global_t(pred, t, t_fr=None):
    return assigned(pred, t, t_fr).select("s1", "id")


def selftest():
    # S1 1: one sure match + one 0.6 candidate; S1 2: a lone 0.55 candidate (empty is worth 0.45); S1 3: nothing
    pred = pl.DataFrame({"s1": [1, 1, 2, 1], "id": [10, 11, 12, 13], "p": [0.99, 0.6, 0.55, 0.52], "country": ["US"] * 4})
    k = decode(assigned(pred, 0.5), 1.0).sort("id")
    assert k["id"].to_list() == [10, 12], k      # 0.6 on a sure cluster is below break-even (~0.7); lone 0.55 > 0.5 is kept
    assert global_t(pred, 0.8)["id"].to_list() == [10]
    fr = pred.with_columns(pl.lit("France").alias("country"))
    assert decode(assigned(fr, 0.5, 0.9), 1.0, 2.0)["id"].to_list() == [10]
    truth = pl.DataFrame({"s1": [1, 2], "id": [10, 12]})
    f = f05(pl.DataFrame({"s1": [1, 2, 3]}), k, truth)
    assert abs(f - 1.0) < 1e-9, f
    print(f"SELFTEST OK (polars {pl.__version__}): assign / decode / France params / f05", flush=True)


def tune(path):
    import polars as pl2   # noqa: F401  (same module; kept for clarity when run locally)
    oof = pl.read_parquet(path).select("s1", "id", "p")
    truth = pl.read_parquet(WORK / "train_gt.parquet")
    s1 = (pl.scan_parquet(WORK / "train_norm2.parquet").filter(pl.col("src") == 1)
          .select(pl.col("id").alias("s1"), pl.col("country").cast(pl.Utf8)).collect())
    oof = oof.join(s1, on="s1")
    base = f05(s1, global_t(oof, T), truth)
    print(f"global t {T}: {base:.5f}", flush=True)
    res = {}
    for fl in FLOORS:
        pa = assigned(oof, fl)
        for g in GAMMAS:
            res[(fl, g)] = f05(s1, decode(pa, g), truth)
            print(f"  floor {fl} gamma {g}: {res[(fl, g)]:.5f} ({res[(fl, g)] - base:+.5f})", flush=True)
    (bf, bg), bF = max(res.items(), key=lambda kv: kv[1])
    by = {c: (f05(s1.filter(pl.col("country") == c), decode(assigned(oof, bf), bg).join(s1.filter(pl.col("country") == c), on="s1", how="semi"),
                  truth.join(s1.filter(pl.col("country") == c), on="s1", how="semi")),
              f05(s1.filter(pl.col("country") == c), global_t(oof, T).join(s1.filter(pl.col("country") == c), on="s1", how="semi"),
                  truth.join(s1.filter(pl.col("country") == c), on="s1", how="semi"))) for c in ("US", "India")}
    print(f"BEST floor {bf} gamma {bg}: {bF:.5f} vs global t {T} {base:.5f} -> delta {bF - base:+.5f} | by country (ef, global) {by}",
          flush=True)


def apply(name, floor, g, floor_fr, g_fr, stages):
    from s15_blend import blended, LOW
    from s6_write import to_tsv
    with timer(f"expected-F decode {name}: floor {floor} gamma {g} | France floor {floor_fr} gamma {g_fr} | stages {stages}"):
        cty = pl.read_parquet(WORK / "test_records.parquet", columns=["id", "country"]).select(pl.col("id").alias("s1"), pl.col("country").cast(pl.Utf8))
        tp = pl.concat([blended(stages, "test", j, LOW) for j in range(BUCKETS)]).join(cty, on="s1")
        m = decode(assigned(tp, floor, floor_fr), g, g_fr)
        ref = global_t(tp, T, 0.90)
        n1 = cty.group_by("country").len("n_s1")
        stat = (m.join(cty, on="s1").group_by("country").len("ef").join(ref.join(cty, on="s1").group_by("country").len("fr90"), on="country")
                .join(n1, on="country").with_columns((pl.col("ef") / pl.col("n_s1")).round(3).alias("ef_per_S1"),
                                                     (pl.col("fr90") / pl.col("n_s1")).round(3).alias("fr90_per_S1")).sort("country"))
        print(stat, flush=True)
        s1_all = (pl.scan_parquet(WORK / "test_norm2.parquet").filter(pl.col("src") == 1).select(pl.col("id").alias("s1")).sort("s1").collect())
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        to_tsv(s1_all, m, "matched_entity_ids", d / "matching_results.tsv")
        print(f"  wrote {d}/matching_results.tsv  matched pairs {m.height:,} (global t 0.80 / France 0.90: {ref.height:,})", flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "selftest":
        selftest()
    elif a[1] == "tune":
        tune(a[2])
    elif a[1] == "apply":
        apply(a[2], float(a[3]), float(a[4]), float(a[5]), float(a[6]), [int(s) for s in a[7:]])
