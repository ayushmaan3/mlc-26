"""Diagnostic for candidate expansion via sibling duplicates (train only, CPU, reads embeddings via memmap).
Question: of the true pairs the v3 candidates MISS, how many look like a sibling the model already predicts?
For every missed true pair (s1, x): max cosine(x, y) over y = the S1's predicted matches (p1_v3 >= t). Also the oracle version
(y = the S1's true matches that ARE in the candidates). Reports the share of missed pairs recoverable at several cosine cutoffs,
and, as the cost side, the same statistic for a sample of non-matching candidates (how many false pairs such a rule would add).
usage: s12_expand_diag.py            (needs train_emb.f16/_ids.npy, train_cand2_*, knn_train_*, p1_v3_train_*, val_stage1_v3.json, train_gt)"""
import json

import numpy as np
import polars as pl

from common import WORK
from s2_block import BUCKETS

DIM = 384


def main():
    t = json.load(open(WORK / "val_stage1_v3.json"))["t"]
    ids = np.load(WORK / "train_emb_ids.npy")
    E = np.memmap(WORK / "train_emb.f16", dtype=np.float16, mode="r", shape=(len(ids), DIM))
    row = pl.DataFrame({"id": ids}).with_row_index("row")
    gt = pl.read_parquet(WORK / "train_gt.parquet")
    pred = pl.concat([pl.read_parquet(WORK / f"p1_v3_train_{j}.parquet", columns=["s1", "id", "p"]).filter(pl.col("p") >= t)
                      for j in range(BUCKETS)])
    pred = pred.sort("p", descending=True).unique("id", keep="first").select("s1", "id")
    missed, found_true, negs = [], [], []
    for j in range(BUCKETS):   # v3 candidates = cand2 + kNN@20, bucket by bucket (low RAM; feat3 not needed)
        c = pl.concat([pl.read_parquet(WORK / f"train_cand2_{j}.parquet", columns=["s1", "id"]),
                       pl.read_parquet(WORK / f"knn_train_{j}.parquet").filter(pl.col("erk") <= 20).select("s1", "id")]).unique()
        g = gt.filter(pl.col("s1") % BUCKETS == j)
        missed.append(g.join(c, on=["s1", "id"], how="anti")); found_true.append(g.join(c, on=["s1", "id"], how="semi"))
        n = c.join(g, on=["s1", "id"], how="anti"); negs.append(n.sample(min(250_000, n.height), seed=j))
        del c, n
    missed, found_true, negs = pl.concat(missed), pl.concat(found_true), pl.concat(negs)
    print(f"true pairs {gt.height:,} | missed by v3 candidates {missed.height:,} ({missed.height / gt.height:.4f})", flush=True)

    def max_sim(pairs, anchors, label):
        """pairs [s1, id]; anchors [s1, id] -> max cos(id, anchor) per pair (anchor != id); NaN if no anchor."""
        a = anchors.rename({"id": "a"})
        x = pairs.join(a, on="s1").filter(pl.col("id") != pl.col("a"))
        x = x.join(row.rename({"row": "rx"}), on="id").join(row.rename({"id": "a", "row": "ra"}), on="a")
        rx, ra = x["rx"].to_numpy(), x["ra"].to_numpy()
        sims = np.empty(len(rx), dtype=np.float32)
        for i in range(0, len(rx), 500_000):
            sims[i:i + 500_000] = (E[np.sort(rx[i:i + 500_000])].astype(np.float32)[np.argsort(np.argsort(rx[i:i + 500_000]))]
                                   * E[ra[i:i + 500_000]].astype(np.float32)).sum(1)
        m = x.select("s1", "id").with_columns(pl.Series("sim", sims)).group_by("s1", "id").agg(pl.col("sim").max())
        r = pairs.join(m, on=["s1", "id"], how="left")
        cuts = (0.85, 0.9, 0.92, 0.95)
        share = {c: round(float((r["sim"] >= c).sum()) / r.height, 4) for c in cuts}
        print(f"  {label}: n={r.height:,} has-anchor {r['sim'].is_not_null().mean():.3f} | share with max cos >= " +
              " ".join(f"{c}:{share[c]}" for c in cuts), flush=True)
        return share

    print("MISSED true pairs vs the S1's PREDICTED matches (realistic expansion):", flush=True)
    max_sim(missed, pred, "missed/pred")
    print("MISSED true pairs vs the S1's true matches found in candidates (oracle upper bound):", flush=True)
    max_sim(missed, found_true, "missed/oracle")
    print("COST: random NON-matching candidates vs the S1's predicted matches (false pairs the rule would admit):", flush=True)
    max_sim(negs, pred, "negatives/pred")


if __name__ == "__main__":
    main()
