# TRACKER: findings, submissions, current work
One page to avoid confusion. START WITH RESUME_HERE.md. Updated 2026-09-26 ~12:00 UTC. Deadline: 2026-09-28 00:00 IST (2026-09-27 18:30 UTC).
Metric: macro F0.5 per S1. Public LB = part of test; private LB decides the final ranking.
Target seen on the LB: ~0.987 to be competitive (>500 teams above 0.97).

## Leaderboard submissions
| # | file | what | our OOF (US+India) | public LB |
|---|---|---|---|---|
| 1 | output (v1) | LightGBM, blocking recall 0.933 | 0.9159 | 0.906 |
| 2 | output_s1 | v2 stage 1 (translit + keys, XGBoost) | 0.9634 | 0.952 |
| 3 | output_s1_ef | v2 s1 + expected-F decode | 0.9634 | 0.953 |
| 4 | output_v2 | v2 stage 2 (stacked on p1) | 0.9661 | 0.948 (worse: stacking fragile) |
| 5 | output_v3s1 | **v3 stage 1** (keys + e5 kNN, recall 0.978, cluster-support feats), t=0.625 | **0.9745** | 0.961 |
| 6 | output_v3s1_t80 | v3 s1 at threshold 0.80 | 0.9731 | 0.963 |
| 7 | output_fr90 | v3 s1, t 0.80, France 0.90 | - | 0.964 |
| 8 | output_v3s5_fr90 | **v3 stage 5** (s1 feats + CE + address/house-number add-on), half-data training, t 0.80 / France 0.90 | **0.9841** (US 0.9867, India 0.9803) | 0.97735 |
| 9 | blend56ft_fr90 (KAGGLE_runC) | **run C**: + fine-tuned e5 kNN candidates (recall 0.9937), full-data stage 5+6 blend, t 0.80 / France 0.90 | **0.98657** (US 0.9873, India 0.9855) | 0.979638 |
| 10 | blend78ft_fr90 (KAGGLE_runD) | **run D**: C union + competition features, stage 7+8 blend, t 0.80 / France 0.90 | **0.98722** (at 0.80: 0.98698; US 0.9878, India 0.9864) | 0.98135 |
| 11 | blend78ft_t80 (KAGGLE_variantD) | same as #10, France threshold 0.80 instead of 0.90 | same | 0.981161 |
| 12 | blend910_fr90 (KAGGLE_runE) | **run E**: D + fold-wise fine-tuned e5 cosine features, stage 9+10 blend, t 0.80 / France 0.90 | **0.98744** (at 0.80: 0.98726; US 0.9879, India 0.9868) | 0.981678 |
| 13 | stack7 (KAGGLE_stack) | set-level XGB stacker on run D p7 (per model), t 0.80 / France 0.90 | **0.98762** at 0.80 (US 0.98819, India 0.98676) | (pending) |
| 14 | stack7ce (KAGGLE_runX) | stack7 + fold-wise mmarco cross-encoder logit on the uncertain band | **0.98835** at 0.80 (US 0.98851, India 0.98811) | **0.983021 (final, best)** |
| - | A: v3s5full(_fr90) | stage 5 FULL data (Kaggle run A) | 0.98445 (US 0.9869, India 0.9807) | not yet |
| - | A: blend56(_fr90) | stage 5 + stage 6 averaged | 0.98454 (US 0.987, India 0.9809) | held as fallback (user's choice) |
| - | out_v3s1_t65 | decoy-sim threshold 0.65 | 0.9742 | not submitted (sim did not reproduce test) |

## Findings (measured)
- CV->LB gap: -0.010 (v1), -0.011 (v2), -0.013 (v3). Validation covers only US+India; France is 15% of test.
- France is NOT visibly broken (matches/S1 3.56 like US; empty-S1 share 5.1%).
- **Test has ~20% more S2+S3 records per S1 than train** (train 4.67, test 5.5-5.8) -> more decoys on test -> a stricter threshold helps
  (t 0.625 -> 0.80: LB +0.002). Dropping S1 in validation does NOT reproduce this (best t only 0.625 -> 0.65).
- GT structure: each matched S1 has 1-5 S2 and 1-6 S3 duplicates; 85% have both sources; 73-75% of S2/S3 records are matched in train.
- Blocking recall: keys 0.9503 -> keys + e5 kNN@20 **0.9784** (+12.8 pairs/S1). kNN@20 alone 0.958.
- Sibling expansion (missed pairs similar to a predicted sibling): only 21.8% of misses at cos>=0.92 -> ceiling ~0.983; parked.
- Per-entity expected-F decoding: no gain over a global threshold.
- **Error budget of v3 stage 1 (OOF, t=0.625):** precision 0.9924, pair recall 0.944.
  | error | count | F0.5 gain if fixed |
  |---|---|---|
  | FN_cand (true pair in candidates, rejected) | 262,406 | **+0.0123** |
  | FN_block (true pair not in candidates) | 165,070 | +0.0077 |
  | FP_decoy | 39,271 | +0.0035 |
  | FP_single | 3,330 | +0.0014 |
  | FP_owned | 12,642 | +0.0012 |
- FN_cand details: 96% unassigned (not stolen by another S1); p quantiles 10/25/50/75/90% = 0.015/0.076/0.247/0.448/0.563;
  42% have p in [0.3, t); 20% p < 0.05. Sample: half have an EMPTY S2/S3 address (exact same name, p ~0);
  others have corrupted house numbers (1161 vs 161, 85 vs 15) or native script + number noise.
- Empty-address FN_cand (31.7% of FN_cand with key features): exact name 84.5%, but name unique among S1 only 6.9% -> genuinely ambiguous.
  Rule "empty addr + exact unique name" tested on OOF: 2,433 pairs, precision 0.257, F0.5 -0.0001 -> REJECTED.
- Non-empty FN_cand: first house number differs in 72%; 45% of those are 1 digit edit away -> number-noise features are the next fix.
- **Name reuse / ceiling (train):** 49.4% of S1 names are shared by >=2 S1 in the same country. 1.996% of true pairs are
  empty-address + shared-name = unresolvable from record contents. Oracle ceiling with only those missed: **F0.5 0.9939**.
  Top LB teams ~0.990 -> near-perfect on everything resolvable. Our resolvable gap ~0.019 -> ADDRESS matching under
  house-number noise is the main lever (stage 5 = first model built for it).
- No ID leak (s1 vs matched id correlation 0.0017; no leading zeros in any raw ID). France-only threshold 0.90 -> LB 0.964 (+0.001).
  Per-source thresholds (S2/S3): no OOF gain.
- India 0.9676 vs US 0.9790 (v3 s1).

- **Stage 5 error budget (t=0.70):** precision 0.9969, recall 0.9594. fix FN_block +0.0073, FN_cand +0.0062 (was +0.0123),
  FP_owned +0.0012, FP_decoy +0.0008, FP_single +0.0006. CV->LB gap shrank to -0.007. Top gains include num_contain, num_diff.
  Stage 5 needed half-data training (BER_TRAIN_BUCKETS=2, EASY_KEEP 20, NPIECE 8) - full data OOM-killed the 12.7 GB runtime.
- LB target moved: ~0.988 needed (rank ~480 at 0.977).
- Address-key blocking (house number + street word, no top-60 cap): catches 8.3% of the resolvable blocking misses for +8.7 pairs/S1 -> dropped.
- Exact-address sibling expansion: 1.24M new pairs, precision 0.007, recovers 5.0% of misses -> dropped.
- Per-record competition features (s16, similarity-based): sample shows true pairs rank 1.01 among a record's candidates (vs 4.28), address gap 0 (vs 27).
- Moved compute to Kaggle (T4 x2, 31 GB RAM, 4 CPUs, background runs up to 12 h). Datasets mlc26-core/emb/v3 uploaded (47/21/52 files).

## Current work (in order)
1. [DONE] stage 5 -> 0.9841 OOF / 0.977 LB.
2. [DONE 20:06 UTC] Kaggle run A (KAGGLE_run.ipynb): feat3 rebuild -> FULL-data stage 5 -> v3s5full(_fr90); stage 6 (511 leaves) -> blend56(_fr90).
3. [DONE 20:42 UTC: v3s7 0.98483, blend78 0.98490 OOF (+0.0006 at t 0.80 vs A)] Kaggle run B (KAGGLE_runB.ipynb): feat3 -> s16 competition feats -> stage 7 full data -> v3s7(_fr90); stage 8 -> blend78(_fr90).
   Pick uploads by the OOF line in each run's log.
3b. [DONE, GO] FT_check.ipynb (code_v2/s17_ft_retrieval.py), Fable idea #1: fine-tune e5 on 300k train pairs (in-batch negatives,
   1 epoch, 2343 steps, 10 min on T4). Held-out 100k S1 (bucket 7, 366,635 true pairs) vs a 1.17M-record db, within-country kNN:
   recall@5/10/20 original 0.9135 / 0.9693 / 0.9771 -> fine-tuned 0.9387 / 0.9943 / 0.9970 (+0.0199 @20).
3c. [DONE 01:16 UTC: recall 0.9937, blend56ft OOF 0.98657 -> upload blend56ft_fr90] KAGGLE_runC.ipynb: fine-tuned e5 (dataset mlc26-ftmodel) encodes train+test (one per GPU) -> kNN@20 -> knnft_* added to the
   union (s10_v3.knn_pairs) -> cos/feat3 -> addr4 rebuilt for the new union -> stage 5 -> v3s5ft(_fr90); stage 6 -> blend56ft(_fr90).
   The fine-tuned model only ADDS candidates: all features (emb_cos, erk, windows) still come from the original e5, because the
   fine-tuned model saw 300k train pairs (4% of positives) and its scores would look better on train than on test.
   ce is null for the new pairs (CE not rerun). knnft_* are saved to state/knnft for reuse.
3d. [BUILT 2026-09-27] KAGGLE_runD.ipynb: C's union (knnft_* from C's Output, no re-encode) + addr4 + s16 idcomp -> stage 7 -> v3s7ft(_fr90); stage 8 -> blend78ft(_fr90).
4. Final day: pick 2 final submissions (best OOF + LB trend), Documentation_template.md, code zip (output/ + code/ + README + requirements).

## Operating rules learned the hard way
- Colab: 12.7 GB RAM shared with the kernel; launch with setsid; one launch only; stop workers with pattern 'python3? s[0-9]'.
- Drive quota counts against the account that CREATES a file (helpers ~15 GB) -> V3 saves only small essentials.
- Big joins over 139M pairs OOM a 12 GB runtime: always process per bucket.
- Validator OOMs on the 108M-pair candidate file -> validate matching_results only (candidates optional).
- Keep the laptop plugged in/awake; a power cut killed a whole night of work once.
