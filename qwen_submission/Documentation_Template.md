# Experiment Log

## Problem (fill at hour 0 — copy from official text, don't paraphrase)
- Task: Business Entity Resolution — for each S1 record list matching S2/S3 ids (see student_resource/README.md, verbatim)
- Metric: macro F0.5 per S1, singletons count (empty pred on singleton = 1.0)
- Submission format: matching_results.tsv (+ candidate_pairs.tsv in final zip), tab-separated, validator in utils/
- Rules: final model MIT/Apache-2.0 and <=8B params; no external APIs/lookups; country open set (France only in test)
- Data: see PLAN.md "Measured facts" (train 2.2M/5.0M/5.3M rows, test 1.7M/4.9M/5.1M)

## Submissions (15 max)
| # | day | exp id | CV | LB | notes |
|---|---|---|---|---|---|
| 1 | 1 | exp03 | 0.9159 | ? | LGB v1, t=0.65, top-50 blocking; validator PASS (--check-ids) |

## Experiments
<!-- one block per run. Results = measured numbers only. -->
### exp01 — blocking v1 (core name exact + number x addr token)
- where: laptop | 2026-09-25 ~00:15 | cost 0
- result (measured, train): 78.5M pairs, 35.6/S1, recall 0.8623
- misses: house numbers off by a digit, generic names shared by 50+ S1 (block cap), native-script names

### exp02 — blocking v2 (+ rare name tok x rare addr tok, rare addr pair, rare name pair; caps 20/100)
- where: laptop | ~00:20–00:35 | cost 0
- result (measured, train): 182.7M unique pairs (83/S1), recall 0.946 uncapped; ranked by weighted shared-key count: recall@10 0.887, @20 0.914, @30 0.923, @50 0.933
- kept top-50 per S1 -> recall ceiling 0.933
- incident: merge of all pairs at once -> RAM 0 GB. Fixed by merging per S1 bucket (8 buckets).
- incident: I killed all python.exe via taskkill /IM (3 procs) — use PID only.
- test blocking: 89.0M pairs, 40.3/S1 (train, top-50 cap); recall ceiling 0.9331

### exp03 — LightGBM v1 (27 features: key counts, rank, rapidfuzz name/addr, number overlap, lengths)
- where: laptop | ~00:50–01:05 | cost 0
- train: bucket 0 (11.1M pairs, 8.0% pos), 300 rounds, lr 0.1, 127 leaves — 258s
- val (buckets 6+7, assignment decode: each S2/S3 -> best S1): **macro F0.5 0.9159 @ t=0.65** (0.9139 @0.5, 0.8982 @0.9)
- caveat: val assignment only sees S1 in val buckets (optimistic); rank/ncand features depend on blocking density (test has fewer S1)
- top gain: len_addr2, rk, ncand_raw, pair_sum, addr_tset, gap_s1_best
- test output: 5.36M matched pairs; S1 with >=1 match 1,614,196/1,732,544. Empty rate / avg matches: France 5.4%/3.28, India 8.0%/2.91, US 5.9%/3.25
- validator: PASS with --check-ids (both files)

## Hypotheses / ideas (unverified)
-

## Incidents (crashes, OOM, lost time) — and the fix
-
