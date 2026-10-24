"""Run R: decode-time rules for generic-name false positives (Fable review 3), validated on labelled train OOF before use.
France looks precision-limited on generic names (shared by several S1); US/India S1 with a shared core name are France in
miniature WITH labels. A rule may be used only if, on train OOF at the submitted setting (t 0.80), it raises macro F0.5 and the
pairs it drops are mostly false (precision < DROP_PREC_MAX; dropping a set helps F0.5 when its precision < ~0.78).
  oof           : OOF p7 on train from run D's saved stage-7 models (xgb_s7_v3_f*.json) -> must reproduce D's OOF (0.98702)
  rules         : every rule variant on OOF -> table + WORK/rules.json (best passing variant per rule family, then combined)
  apply <name> <rules.json> <stage...> : test p of the stages (mean), t 0.80 / France 0.90, passing rules -> OUT/<name>/
Rule families (generic = the S1's core name is shared by >= 2 S1 of its country, norm3 core_n1 >= 2):
  num  : generic, both have house numbers, none in common, p < k      -> drop
  noad : generic, the S2/S3 record has no address, p < k              -> drop
  cap  : more matches for one S1 than the train GT ever has (q-quantile) -> keep the top by p
  amb  : the record's top-2 S1 (all candidates) share the same generic core name and p1 - p2 < d -> the record gets no match"""
import json
import sys

import numpy as np
import polars as pl

from common import OUT, WORK, done, f05, timer
from s2_block import BUCKETS
import s4_train as s4
from s15_blend import blended, LOW
from s6_write import to_tsv

T, T_FR = 0.80, 0.90
DROP_PREC_MAX = 0.70
LOWP = 0.5   # pairs below this can never be assigned at t 0.80 and are too far below p1 to matter for 'amb'
GRID = {"num": [0.97, 0.99, 1.01], "noad": [0.97, 0.99, 1.01], "cap": [0.99, 0.999], "amb": [0.05, 0.10]}


def oof():
    import xgboost as xgb
    bsts = [xgb.Booster(model_file=str(s4.model_path(7, f))) for f in range(2)]
    for b in bsts:
        b.set_param({"device": "cuda"})
    cols = bsts[0].feature_names
    have = s4.feature_cols(7)
    assert cols == bsts[1].feature_names and sorted(cols) == sorted(have), f"feature mismatch: {set(cols) ^ set(have)}"
    print(f"stage 7 models from run D: {len(cols)} features, all present", flush=True)
    for f, va in enumerate(s4.FOLDS):
        for j in va:
            if not done(s4.p_path(7, "train", j)):
                with timer(f"oof stage 7 bucket {j}"):
                    s4.predict([bsts[1 - f]], "train", j, 7, cols).write_parquet(s4.p_path(7, "train", j))
    s4.evaluate(7)
    (pl.concat([pl.read_parquet(s4.p_path(7, "train", j), columns=["s1", "id", "p"]).filter(pl.col("p") >= LOWP) for j in range(BUCKETS)])
     .write_parquet(OUT / "state" / "oof_p7_ge05.parquet"))


def oof_pred():
    """OOF p7 >= LOWP: from WORK (this run) or from a previous run R's Output (state/oof_p7_ge05.parquet linked into WORK)."""
    if all(done(s4.p_path(7, "train", j)) for j in range(BUCKETS)):
        return pl.concat([pl.read_parquet(s4.p_path(7, "train", j), columns=["s1", "id", "p"]).filter(pl.col("p") >= LOWP)
                          for j in range(BUCKETS)])
    return pl.read_parquet(WORK / "oof_p7_ge05.parquet")


def context(split):
    n = pl.read_parquet(WORK / f"{split}_norm3.parquet", columns=["id", "src", "country", "core", "core_n1", "nums", "addr_n"])
    s1 = n.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), pl.col("country").cast(pl.Utf8), pl.col("core").alias("s1_core"),
                                             pl.col("core_n1").alias("gen"), pl.col("nums").alias("s1_nums"))
    rec = n.filter(pl.col("src") != 1).select("id", pl.col("nums").alias("r_nums"),
                                              (pl.col("addr_n").fill_null("").str.strip_chars() == "").alias("r_noaddr"))
    return s1, rec


def decode(pred, t_fr=None):
    """Same as s4_train.assign / s15_blend _fr90: keep p >= t (France: t_fr), each record -> its best S1. Keeps p."""
    q = pred.filter(pl.when(pl.col("country") == "France").then(pl.col("p") >= (t_fr or T)).otherwise(pl.col("p") >= T))
    return q.sort("p", descending=True).unique("id", keep="first")


def ambiguous(pred, s1ctx):
    """Records whose two best S1 (by p, all candidates >= LOWP) share the same generic core name: [id, gap]."""
    top = (pred.sort("p", descending=True).group_by("id", maintain_order=True).head(2)
           .join(s1ctx.select("s1", "s1_core", "gen"), on="s1")
           .sort(["id", "p"], descending=[False, True])
           .group_by("id").agg(pl.len().alias("k"), pl.col("p").first().alias("p1"), pl.col("p").last().alias("p2"),
                               pl.col("s1_core").first().alias("c1"), pl.col("s1_core").last().alias("c2"),
                               pl.col("gen").first().alias("g1")))
    return (top.filter((pl.col("k") == 2) & (pl.col("c1") == pl.col("c2")) & (pl.col("g1") >= 2))
            .select("id", (pl.col("p1") - pl.col("p2")).alias("amb_gap")))


def drop_mask(fam, v, a, amb, cap_n):
    """Boolean expression over assigned pairs a (with context) = pairs this rule drops."""
    gen = pl.col("gen") >= 2
    if fam == "num":
        both = (pl.col("s1_nums").list.len() > 0) & (pl.col("r_nums").list.len() > 0)
        return gen & both & (pl.col("s1_nums").list.set_intersection(pl.col("r_nums")).list.len() == 0) & (pl.col("p") < v)
    if fam == "noad":
        return gen & pl.col("r_noaddr") & (pl.col("p") < v)
    if fam == "cap":
        return pl.col("p").rank("ordinal", descending=True).over("s1") > cap_n[v]
    if fam == "amb":
        return pl.col("amb_gap") < v   # column joined by with_ctx (null = record not ambiguous)
    raise ValueError(fam)


def with_ctx(a, s1ctx, rec, amb):
    return a.join(s1ctx.drop("country"), on="s1", how="left").join(rec, on="id", how="left").join(amb, on="id", how="left")


def apply_rules(a, sel, amb, cap_n):
    m = pl.lit(False)
    for fam, v in sel.items():
        m = m | drop_mask(fam, v, a, amb, cap_n).fill_null(False)   # null (missing context) must never drop a match
    return a.with_columns(m.alias("_drop"))


def rules():
    with timer("rules on train OOF"):
        s1ctx, rec = context("train")
        pred = oof_pred().join(s1ctx.select("s1", "country"), on="s1")
        truth = pl.read_parquet(WORK / "train_gt.parquet")
        s1_all = s1ctx.select("s1", "country", "gen")
        n_true = truth.group_by("s1").len()["len"]
        cap_n = {q: int(np.ceil(n_true.quantile(q))) for q in GRID["cap"]}
        amb = ambiguous(pred, s1ctx)
        a = with_ctx(decode(pred), s1ctx, rec, amb)
        (a.filter(pl.col("p") < 0.99).join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "id"], how="left")
         .select("s1", "id", "p", "country", "gen", pl.col("y").fill_null(0))
         .write_parquet(OUT / "state" / "oof_band.parquet"))   # labelled uncertain matches (for an LLM-veto check)
        subsets = {"all": s1_all, "US": s1_all.filter(pl.col("country") == "US"), "India": s1_all.filter(pl.col("country") == "India"),
                   "generic": s1_all.filter(pl.col("gen") >= 2)}

        def score(pairs):
            p2 = pairs.select("s1", "id")
            return {k: f05(v, p2.join(v.select("s1"), on="s1", how="semi"), truth.join(v.select("s1"), on="s1", how="semi"))
                    for k, v in subsets.items()}

        base = score(a)
        print(f"baseline OOF at t {T}: " + " | ".join(f"{k} {v:.5f}" for k, v in base.items())
              + f" | generic S1 {subsets['generic'].height:,} of {s1_all.height:,} | caps {cap_n}", flush=True)
        rows, best = [], {}
        for fam, vals in GRID.items():
            for v in vals:
                x = apply_rules(a, {fam: v}, amb, cap_n)
                d = x.filter(pl.col("_drop")).select("s1", "id")
                prec = d.join(truth, on=["s1", "id"], how="semi").height / max(d.height, 1)
                sc = score(x.filter(~pl.col("_drop")))
                dF = {k: sc[k] - base[k] for k in sc}
                ok = dF["all"] > 0 and prec < DROP_PREC_MAX and d.height > 0
                rows.append((fam, v, d.height, prec, dF, ok))
                print(f"  {fam:5s} {v:<6} dropped {d.height:>8,}  precision {prec:.3f}  dF all {dF['all']:+.5f}  US {dF['US']:+.5f}"
                      f"  India {dF['India']:+.5f}  generic {dF['generic']:+.5f}  {'PASS' if ok else '-'}", flush=True)
                if ok and dF["all"] > best.get(fam, (None, 0))[1]:
                    best[fam] = (v, dF["all"])
        sel = {fam: v for fam, (v, _) in best.items()}
        comb = score(apply_rules(a, sel, amb, cap_n).filter(~pl.col("_drop"))) if sel else base
        print(f"SELECTED {sel} -> combined OOF at t {T}: " + " | ".join(f"{k} {comb[k]:.5f} ({comb[k] - base[k]:+.5f})" for k in comb),
              flush=True)
        json.dump({"selected": sel, "cap_n": {str(k): v for k, v in cap_n.items()}, "baseline": base, "combined": comb,
                   "table": [(f, v, n, p, d, ok) for f, v, n, p, d, ok in rows]}, open(WORK / "rules.json", "w"), indent=1)


def apply(name, rules_json, stages):
    r = json.load(open(rules_json))
    sel = r["selected"]
    cap_n = {float(k): v for k, v in r["cap_n"].items()}
    with timer(f"apply rules {sel} to stages {stages} -> {name}"):
        s1ctx, rec = context("test")
        tp = (pl.concat([blended(stages, "test", j, LOW) for j in range(BUCKETS)]).filter(pl.col("p") >= LOWP)
              .join(s1ctx.select("s1", "country"), on="s1"))
        a = with_ctx(decode(tp, T_FR), s1ctx, rec, ambiguous(tp, s1ctx))
        x = apply_rules(a, sel, None, cap_n)
        n_s1 = s1ctx.group_by("country").len("n_s1")
        print((x.group_by("country").agg(pl.len().alias("matched"), pl.col("_drop").sum().alias("dropped"))
               .join(n_s1, on="country").with_columns((pl.col("matched") / pl.col("n_s1")).round(3).alias("matches_per_S1"),
                                                      ((pl.col("matched") - pl.col("dropped")) / pl.col("n_s1")).round(3).alias("after_rules"))
               .sort("country")), flush=True)
        m = x.filter(~pl.col("_drop")).select("s1", "id")
        s1_all = (pl.scan_parquet(WORK / "test_norm2.parquet").filter(pl.col("src") == 1)
                  .select(pl.col("id").alias("s1")).sort("s1").collect())
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        to_tsv(s1_all, m, "matched_entity_ids", d / "matching_results.tsv")
        print(f"  wrote {d}/matching_results.tsv  matched pairs {m.height:,} (before rules {a.height:,})", flush=True)


def selftest():
    s1ctx = pl.DataFrame({"s1": [1, 2, 3], "country": ["US", "US", "France"], "s1_core": ["acme", "acme", "zeta"],
                          "gen": [2, 2, 1], "s1_nums": [["12"], ["40"], []]})
    rec = pl.DataFrame({"id": [10, 11, 12, 13], "r_nums": [["12"], ["99"], [], ["5"]], "r_noaddr": [False, False, True, False]})
    pred = pl.DataFrame({"s1": [1, 2, 1, 1, 3, 2], "id": [10, 10, 11, 12, 13, 13], "p": [0.95, 0.93, 0.85, 0.9, 0.97, 0.6],
                         "country": ["US", "US", "US", "US", "France", "US"]})
    amb = ambiguous(pred, s1ctx)
    assert amb.sort("id")["id"].to_list() == [10] and abs(amb["amb_gap"][0] - 0.02) < 1e-9, amb
    a = with_ctx(decode(pred, T_FR), s1ctx, rec, amb)
    assert sorted(a["id"].to_list()) == [10, 11, 12, 13], a
    cap_n = {0.99: 1}
    got = {fam: sorted(apply_rules(a, {fam: v}, None, cap_n).filter(pl.col("_drop"))["id"].to_list())
           for fam, v in (("num", 1.01), ("noad", 1.01), ("cap", 0.99), ("amb", 0.05))}
    assert got == {"num": [11], "noad": [12], "cap": [11, 12], "amb": [10]}, got
    x = apply_rules(a, {"num": 1.01, "amb": 0.05}, None, cap_n)
    assert x["_drop"].null_count() == 0 and sorted(x.filter(~pl.col("_drop"))["id"].to_list()) == [12, 13]
    truth = pl.DataFrame({"s1": [1, 3], "id": [10, 13]})
    f = f05(s1ctx.select("s1"), x.filter(~pl.col("_drop")).select("s1", "id"), truth)
    print(f"SELFTEST OK (polars {pl.__version__}): ambiguous / decode / 4 rule families / f05 = {f:.4f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "oof":
        oof()
    elif sys.argv[1] == "rules":
        rules()
    elif sys.argv[1] == "apply":
        apply(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4:]])
