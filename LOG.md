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
| 4 | 2 | v2 stage 2 (stacked on p1, comp features) | 0.9661 (t=0.675) | **0.948** | OOF +0.0027 but LB -0.004 vs stage 1 -> stacking fragile on test |
| 3 | 1 | v2 stage 1 + s9 expected-F decode | 0.96336 | **0.953** | decode ~tied with global t on OOF (-0.00003); LB +0.001 = noise level |
| 2 | 1 | v2 stage 1 (exp04+) | **0.96339** (t=0.65) | **0.952** | CV-LB gap -0.011 (v1: -0.010) | XGB 2-fold OOF, 59 feats, blocking v2 (recall 0.950). Rank ~300 (many teams 0.969-0.971). |
| 1 | 1 | exp03 | 0.9159 | **0.906** (0.90557) | LGB v1, t=0.65, top-50 blocking; validator PASS. Rank ~100 at the time. |

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

### exp04 — v2 normalize + blocking (Colab CPU, 2 vCPU / 12 GB)
- where: Colab T4 runtime | 2026-09-25 ~12:00-14:00 IST | cost 0
- error analysis of exp03 (val buckets 6,7): India F 0.885 vs US 0.937. Blocking misses 127.7k: 33% non-Latin S2/S3 name, 23% empty addr, 12% web/handle name. True pairs below t: 122.9k (17% non-Latin, 22% empty addr). FP 74.6k, of which 32k have their true S1 outside the val buckets (val-decode artefact) -> use global OOF decode.
- s1 v2: anyascii transliteration (+ anusvara m->n), phonetic skeleton (skel), no-space name (nosp). 470 s/split.
- s2 v2: +4 key types (skel exact, nosp exact, rare skel tok x rare addr tok, rare skel tok x number), TOPK 60, keys in 16 id-chunks (4 chunks OOM-killed the Colab kernel).
- result (measured, train): 110.9M pairs, 50.2/S1, **recall 0.9503** (was 0.933). Bucket 0 uncapped 0.9633; @10 0.8905 @20 0.924 @30 0.9357 @40 0.9424 @50 0.9468 @60 0.9501.

## Status 2026-09-25 evening (before travel)
- Target: LB >= 0.96 (0.97 to be competitive). CV->LB gap on exp03: -0.010.
- v2 features: prep done for train+test (norm3, idf). **Buckets done: train 0-7 (all), test 0-2. Left: test 3-7** (~8 min each).
  Driver stopped cleanly by PID at 11:27 UTC during test 3; _tmp_ files removed. All 36 v2 artifacts verified on Drive work_v2 (sizes match, 15 GB).
  Measured bucket times: train 556-672 s, test 353-479 s.
  Measured per bucket: ~7.5 min, peak RSS 8.3 GB (spilling version). test_feat2_0/1 verified identical to the non-spilling code.
- Next session plan: (1) RESUME.md cells 1-3 -> finish features; (2) stage 1 -> OOF F0.5 -> write + submit a stage-1 file right away
  (run_rest.sh now does this automatically -> MLC'26/output_s1); (3) comp + stage 2 -> submit; (4) embeddings (e5-small) as blocking + feature.

## Models (licence verified via HF API 2026-09-25)
| model | licence | params | use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 117.7M | chosen: fast multilingual name/addr embeddings (needs "query: " prefix) |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 117.7M | chosen: backup/second view |
| BAAI/bge-m3 | MIT | ~568M | skipped: too slow for ~22M strings on a T4 |
| intfloat/multilingual-e5-base | MIT | 278M | not downloaded |
| sentence-transformers/LaBSE | Apache-2.0 | 471M | not downloaded |
Local copy: D:/Amazzon/models/<name> (492 MB / 477 MB). Drive copy: MLC'26/models/<name> (snapshot_download, offline-load verified on Colab).
Quick test (mean-pooled cosine; e5 uses "query: " prefix), measured:
- e5-small: Laxmi Mgmt en~hi 0.896 | Premier Eng en~odia 0.881 | FR accent~plain 0.947 | unrelated (Laxmi~Holloway) 0.796 -> cross-script works, but scores are compressed, so use as a ranked/relative feature. T4 fp16 ~6.1k names/s with the CPU busy (so ~1 h for ~22M strings).
- MiniLM-L12: 0.544 | 0.300 | 0.676 | 0.162 -> better spread but weak cross-script. ~7.6k names/s.
- Decision: e5-small first (blocking by ANN + cosine feature); MiniLM as optional second view.
- Added for plan v3 (licences verified via HF API): intfloat/multilingual-e5-base (MIT, 278.0M) and cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0, 117.6M).
  Offline load verified on Colab T4. e5-base: en~hi 0.879, unrelated 0.791 (same compression as small).
  Cross-encoder zero-shot: "Laxmi Management Pvt Ltd | Faridabad" vs Hindi-script same entity = +7.25 logit, vs unrelated US business = -9.25 -> strong cross-script signal before any fine-tuning.
- Wheels for offline install on Drive MLC'26/wheels: rapidfuzz 3.14.6, anyascii 0.3.3, faiss-cpu 1.15.1 (cp313).
- Rank dropped to ~180 at LB 0.906; need >= 0.972. Full plan: PLAN_v3.md.

## 2026-09-25 evening: notebooks for unattended runs (compute on account 1 exhausted)
- MAIN_pipeline.ipynb (your account): restore -> run_rest (test feats 3-7 -> stage 1 -> output_s1 -> comp -> stage 2 -> output_v2)
  + run_ce (waits for stage 1 + comp; fine-tunes the mmarco cross-encoder per fold on uncertain pairs p1 in [0.03, 0.97], max 1.5M/fold;
  scores OOF train + test; then stage 3 = stage 2 + ce (+ emb_cos if friend's 16 embcos files exist) -> output_s3).
- PARALLEL_embeddings.ipynb (friend's account): e5-small encode -> within-country GPU kNN top-20 -> recall report -> cosine for all cand2 pairs.
  Writes only MLC'26/work_emb.
- New code (code_v2 locally; the notebooks write it to /content/ber/src and code_v3 on Drive): s8_ce.py, e1_emb.py, s4_train.py stage 3
  (joins ce_* / embcos_* when present; real feature names in XGBoost).
- Verified locally end to end on a real-data sample (3.2k S1 per split, ~29k records): s1 -> s2 -> s3 -> s4(1) -> s6(1) -> s5 -> s4(2) -> s8 -> e1 (all steps) -> s4(3) -> s6(3), all ran.
  Sample numbers are not meaningful (easy sample: OOF 0.99; kNN@10 union recall 0.995).

## New: s9_decode.py (per-S1 expected-F0.5 decoding, tuned on OOF; prints VERDICT). Smoke: no gain on the easy sample (-0.0003), so it only replaces the global threshold if it beats it on OOF.

## s4 v3 (for next account): stage >= 2 training keeps p1 >= 0.01 + 10% hash-sample of easy negatives (weight 10); comp/extra joins filtered per quarter; threshold grid up to 0.95. Smoke: stage-2 OOF unchanged (0.98957), all positives kept, 64% of rows dropped. Stage 2 on account 2 was OOM-killed during data build (unfiltered comp join) -> this fix.

## France check (test p1): mean best p FR 0.954 / IN 0.935 / US 0.946; share matched 0.948/0.931/0.941; matches/S1 3.3/3.1/3.3 -> France not visibly worse; gap likely spread. Uncertain FR pairs: same street, house no. off by 1-2, generic name word swaps, legal-form swaps, acronyms (CM), region vs department.
## Friend run (account 3): encode train 8242 s (2 vCPU tokenizer-bound), knn India 594 s; recall step OOM (all 110M pairs at once) -> bucketed recall (smoke-identical numbers).

### exp05 — embedding kNN blocking test (friend's account, PARALLEL_embeddings.ipynb) — 2026-09-25 18:15 UTC
- e5-small ("query: name | address"), fp16, max_len 64. Encode train 12.53M records: 8242 s (2 vCPU tokenizer-bound, ~1.5k/s; not the GPU).
- kNN within country on the T4 (tiled matmul, top-20): India 594 s, US ~10 min.
- **result (measured, train):**
  | | recall | new pairs/S1 |
  |---|---|---|
  | cand2 (keys v2) | 0.9503 | 0 |
  | kNN@5 alone / union | 0.8554 / 0.9664 | 0.9 |
  | kNN@10 alone / union | 0.9457 / **0.9744** | 4.3 |
  | kNN@20 alone / union | 0.9582 / **0.9784** | 12.8 |
- kNN@20 alone beats all hand-built keys; the union cuts the unreachable true pairs from 4.97% to 2.16%.
- Saved on Drive MLC'26/work_emb: train_emb.f16 (9,620,766,720 B) + train_emb_ids.npy, knn_train_0..7.parquet, recall_report.txt.
- Not done: embcos (cos step far too slow: random memmap reads, ~1 h for less than 1 bucket), all test-side steps.
- Incidents: the recall step loaded all 110M pairs -> OOM -> bucketed recall; a duplicate job + duplicate cp after the relaunch (killed by PID).

### v3 code (2026-09-26 ~00:30 local), verified locally, not yet run at scale
- s10_v3.py: union = cand2 + kNN@20; s3 features for the NEW pairs only; feat2 rows reused unchanged (verified identical);
  emb_cos for all pairs (sequential memmap reads: pairs sorted by row, S1 rows in RAM); erk, in_keys, in_knn, emb_rk, emb_gap, ncand3;
  per-S1 windows recomputed over the union in S1 quarters. -> {split}_feat3_{j}.
- Versioning: BER_TAG (_v3) on p/xgb/comp/ce/val files, BER_FEAT (feat3). Empty tag = v2 names, so the v2 run is unaffected.
- The notebooks now write ALL code files themselves (the old common.py on Drive lacks TAG and would crash the new s4).
- Local end-to-end run of the real run_v3.sh against a fake /content + fake Drive (sample test set built from smoke records):
  exit 0, feat3 recall 0.9963 (= kNN@20 union), v3 OOF s1 0.99322 / s2 0.99240 / s3 0.99330 (v2 s1 on the same sample 0.9895),
  3 submissions each validator PASS; 107 artifacts synced; re-run = 0 recomputed steps, no re-submission.
  (Sample numbers are optimistic; only the relative gain and the mechanics are meaningful.)
- Notebooks: PARALLEL_embeddings.ipynb is now TEST-SIDE ONLY (encode test ~2.2 h + kNN test); V3_pipeline.ipynb = restore -> cos -> feat3 -> s1/s2/CE/s3 with submissions output_v3s1/2/3.

### Night of 25->26: power cut killed both runtimes
- Survived on Drive (saved by the scripts as they went): stage 2 models, p2 (16), val_stage2.json, output_v2 (validator passed before the copy). Lost: CE (not started), test encoding (~1 h in, no checkpoint).
### Stage-2 LB drop diagnosis (share of pairs with 0.05 < p < 0.95)
- stage 1: train OOF 1.13% vs test 1.66% (+47%); stage 2: 0.88% vs 1.48% (+68%). Test is less certain, and stacking amplifies it.
  Cause (probable): test p1 = mean of 2 fold models (smoother) vs single-model OOF on train; plus test density differs (1.73M vs 2.21M S1).
- Fixes (smoke-verified): (1) fold-consistent stacking: test p/ce keep per-model columns (p_m0/p_m1, ce_m0/ce_m1); comp for test built per model
  (comp_*_test_m{0,1}); stacked booster g reads model 1-g's inputs. (2) New stage 4 = stage-1 features + CE logit, NO comp (safe default).
  (3) e1 encode: 8 chunks, each copied to Drive when done (BER_EMB_SYNC); resumes from finished chunks; identical embeddings (max diff 0.0).
- V3 order now: stage 1 -> CE -> stage 4 (submit) -> comp/stage 2/stage 3 (experiments). Local full run_v3.sh: exit 0, 4 submissions PASS.

### GT structure (train) -> cluster-support features
- A matched S1 has 1-5 S2 and 1-6 S3 duplicates (most common n2,n3: 1,1 / 1,2 / 2,1 / 2,2); 85.2% have both S2 and S3. 73-75% of S2/S3 records are matched.
- New in s10 cos step: anchors = each S1's top-3 candidates by emb_cos; per candidate sup_max / sup_mean / sup_n90 = similarity to the S1's other anchors
  (no learned scores -> train/test consistent). Brute-force check identical. Smoke: mean sup_n90 1.89 (true) vs 0.03 (false).
  Smoke OOF with support: v3 s1 0.99331 (was 0.99321), s4 0.99402 (was 0.99363).
- Atomic Drive saves in run_v3 (copy to .part, then rename); restore drops *.part.
- Helper mode: V3 cell 4c (BER_V3_MODE=test) builds only the test half of feat3 on a second account; the main run fetches the finished buckets
  from Drive before building test. Local test: helper built 8 test buckets, main rebuilt none, 4 submissions PASS.

### Drive quota gotcha (2026-09-26 ~04:06 UTC)
- "Google Drive storage quota has been exceeded" on a helper account: files count against the account that CREATES them, even inside the
  owner's shared folder (owner has 400 GB; helpers ~15 GB). The FUSE mount may show full sizes before the cloud upload succeeds.
- Mounting the owner's Drive from a helper Colab is NOT possible (no account choice offered). So: notebooks print this account's free space; V3 saves only small essentials
  (models, val json, p1_v3, ce_v3, helper's test_feat3; NOT train_feat3 / p2-p4 / comp). Smoke: 941 MB saved; a second account with only Drive's contents rebuilt 8 train feat3 buckets, retrained nothing, finished all submissions.
- CE models verified on Drive from the MAIN account (470,588,492 B each; that account 12 GB free).
- Rescue cell for live sessions: drive.flush_and_unmount(); drive.mount(force_remount=True) as owner; re-copy test_emb.f16 (.part + mv), knn_test_*.
- CE training done (account 2): fold 1 2080 s at 376 pairs/s, final loss ~0.20; models copied (verify ownership/size on the owner's Drive).

### V3 at full scale (account 5, 2026-09-26)
- feat3 train: 139,149,631 pairs, positives 7,473,295 / 7,638,365 = **recall 0.9784** (matches the sample estimate). Test feat3 buckets ~189 s each, peak RSS ~8.7 GB.
- Incidents: cos test GPU OOM (2M-pair chunks + fragmentation) -> 500k chunks + empty_cache + expandable_segments (identical output).
  Duplicate launches + my stop pattern 'python s[0-9]' missed Colab's 'python3 s...' workers -> orphans held GPU memory; fixed pattern 'python3? s[0-9]' + kill all nvidia-smi compute pids.
  Stage 1 on 139M x 69 was OOM-killed building the QuantileDMatrix -> stage-1/4 training now keeps each S1's top-10 by rk_s1_ps OR emb_rk + 10% hash sample of the rest (weight 10);
  smoke: all positives kept, 56% of rows on the sample, OOF 0.99349 vs 0.99331 unsampled.
- **v3 stage 1 OOF 0.9745 at t=0.625** (v2 s1 0.9634): India 0.9676 (+0.022), US 0.9790 (+0.004). Expected LB ~0.963 at the usual -0.011 gap.
- The official validator was OOM-killed on the v3 candidate file (~108M pairs) -> submit() validates matching_results only (candidates are optional in the validator).

## Incidents on account 2 (2026-09-25 13:00-14:30 UTC)
- Jobs launched with plain nohup died with the kernel -> all launches now use setsid (notebooks rebuilt).
- test bucket 3 killed (OOM, no dmesg access): s3 load_records switched to pyarrow dataset filter + RSS logging. Measured on Colab: load 5.7 GB peak, chunks 7.9 GB peak, ~7.1 GB still available; buckets 3-7 then completed.
- stage 1 training crashed the whole runtime (RAM): pieces were half-buckets converted through a float64 temp copy. Fix: NPIECE=4 quarter pieces + direct Float32 cast (same data, same model). Smoke OOF unchanged (0.9895 vs 0.9896).

## Hypotheses / ideas (unverified)
-

## Incidents (crashes, OOM, lost time) — and the fix
- 2026-09-25 ~06-07: portal upload of exp03 matching_results.tsv (91.5 MB, validator PASS, ASCII/LF/no BOM) returns `Bad Request`, url "". Not a format issue. Suspect upload size limit or portal-side; unresolved.
- 2026-09-25 ~14:30 IST: s3 features v2 OOM x3 on Colab (12 GB cgroup shared with kernel). Causes measured: IDF group_by over 110M exploded tokens with Utf8 country (+4 GB), two joins on the 12.5M-row norm table (copies). Last attempt took the whole runtime down (kernel gone). Fix: s3 split into prep (IDF counted in 8 id-chunks, row-aligned hstack, window freq; standalone peak 7.5 GB) + one process per bucket loading only needed records; s4 predicts per half-bucket; s5 per-id stats via one group_by. Local canonical copy: D:/Amazzon/code_v2.

## 2026-09-26/27: fine-tuned e5 for retrieval (Fable idea #1) -> run C
- s17_ft_retrieval.py on a spare Colab T4: 300k train pairs (S1 buckets 0-6), in-batch negatives, 1 epoch, 10 min.
  Held-out 100k S1 (bucket 7), db 1.17M: recall@5/10/20 0.9135/0.9693/0.9771 -> 0.9387/0.9943/0.9970.
- Run C (KAGGLE_runC.ipynb): fine-tuned model from dataset mlc26-ftmodel; encode train/test one per GPU; kNN@20 -> knnft_*;
  s10_v3.knn_pairs adds them to the union (features still from the original e5); addr4 rebuilt (s14 now imports s10_v3.union_pairs).
- Local rehearsal (smoke_kaggle/run_C_local.sh, fake ftmodel = original e5 so knnft == knn): exit 0, v3s5ft/blend56ft all PASS.
  Local-only artifact: one GPU here, so "GPU 1" fell back to CPU (5 rec/s); both jobs pointed at GPU 0 for the rehearsal.
- Kaggle run A done (20:06 UTC): v3s5full OOF 0.98445 (t 0.675; 0.98410 at 0.80), stage 6 0.98440, blend56 0.98454.
  Full data vs half data +0.0004 only -> candidates (recall) are the bottleneck, not the model. Runs B and C running. See RESUME_HERE.md.

## 2026-09-27 01:16 UTC — Kaggle run C DONE (KAGGLE_runC.ipynb, fine-tuned e5 retrieval + full-data stage 5)
- FT e5 loads (transformers 5.0.0): cos same-business 0.960, different -0.090. Encode train 4337 s, test 4693 s (one per GPU).
- New pairs per bucket: train ~6.48M, test ~4.90M (A had 3.53M / 2.73M).
- **feat3 train: 162,712,539 pairs, recall 0.9937** (A: 139,149,631 pairs, 0.9784) -> +0.0153 blocking recall.
- v3s5ft [5]: OOF best t=0.65 **F0.5=0.98636** | at 0.80: 0.98575 | US 0.9871, India 0.9852 (A: 0.98445, India 0.9807)
- stage 6 alone: t=0.65 F=0.98644
- blend56ft [5, 6]: OOF best t=0.625 **F0.5=0.98657** | at 0.80: 0.98593 | US 0.9873, India 0.9855
- Matched pairs: v3s5ft 5,818,346 / _fr90 5,755,267; blend56ft 5,825,892 / _fr90 5,756,635. All 4 validator PASS.
- Gain vs A: +0.0020 OOF, almost all from India (+0.0045). Logloss got slightly worse (0.0037 -> 0.0050) = harder candidate set, expected.
- Note: the log pasted as "run B" on 2026-09-27 was run A's log (ends "KAGGLE RUN DONE", v3s5full/blend56). Run B results still unseen.

## 2026-09-26 20:42 UTC — Kaggle run B DONE (KAGGLE_runB.ipynb, stage 7 = stage 5 + s16 competition features, A's union)
- feat3 train 139,149,631 pairs, recall 0.9784 (same union as A). idcomp train 134 s, test 117 s. Stage 7: 84 features, full data OK.
- Top gain now idc_rk_em, idc_rk_ps, idc_gap_ps, idc_gap_em (competition features dominate). Logloss 0.0037 -> 0.0032.
- v3s7 [7]: OOF best t=0.725 **F0.5=0.98483** | at 0.80: **0.98475** | US 0.9874, India 0.981   (A v3s5full: 0.98445 / 0.98410)
- stage 8 alone: t=0.75 F=0.98478
- blend78 [7, 8]: OOF best t=0.725 **F0.5=0.98490** | at 0.80: **0.98479** | US 0.9874, India 0.9811 (A blend56: 0.98454 / 0.98419)
- Matched pairs: v3s7 5,781,737 / _fr90 5,756,054; blend78 5,781,359 / _fr90 5,755,384. All PASS.
- Gain vs A: +0.0004 at best t, **+0.0006 at t=0.80** (the submitted threshold); F curve flat from 0.65 to 0.80 -> decoy-robust.
  Competition features target decoys, and test has ~20% more decoys than train -> may gain more on LB than on OOF.
- Decision: build run D = stage 7/8 on run C's union (KAGGLE_runD.ipynb, reuses C's knnft_* from C's Output).
- 2026-09-27: blend56ft_fr90 LB 0.979638 (prev best 0.97735): +0.0023 LB for +0.0020 OOF -> OOF gains transfer ~1:1. User: cut-off now ~0.989.

## 2026-09-27 ~02:45 UTC — hunt for a "missed trick" (top 21 teams all at 0.990 LB); scripts in session scratchpad (ov_*.py)
- Raw counts: train S1 2,206,821 (US 1,323,633 / India 883,188), S2 5,034,616, S3 5,285,603; GT empty rows 123,247 (5.6%).
  test S1 1,732,544 (France 259,452 / India 809,986 / US 663,106), S2 4,887,273, S3 5,082,316.
- S2+S3 per S1: train US 4.67 India 4.68; test US 5.76 India 5.82 France 5.53 -> the extra records are in ALL countries.
- **No train/test leak**: exact (name|address) shared between any test file and any train file: 0 (S1) / <=34 records (S2/S3).
- **No ID or row-order leak**: IDs random 9-digit; corr(row S1, row of its match) 0.001 (S2) / -0.0007 (S3); matched share
  flat 73% across row deciles; siblings never adjacent.
- Decoys: train unmatched S2 have an exact S1 name 2.4% (India) / 4.1% (US) vs matched 7.3% / 17.6%. Test S2 exact-name share
  India 5.3%, US 11.2% -> solving the mix gives test matched share ~0.59 (India) / ~0.53 (US) vs train 0.73. Test decoys are the
  SAME easy type as train, just ~1.6x more of them (no hard-negative injection). France exact-name share 22% (generic names).
- Our test predictions 3.32 matches/S1 = train 3.46 x recall 0.96 / precision 0.997 -> test true-match volume per S1 = train.
- Inference (NOT measured): if US+India test ~= OOF 0.9866, LB 0.9796 implies France ~0.94 (France = 15% of test S1).
  Proposed LB probe: blend56ft_fr90 with France rows blanked -> France F = s_F + (0.97964 - LB_probe) / 0.1497 (s_F ~0.05).

## 2026-09-27 ~04:15 UTC — run E built + rehearsed (Fable review 2: fold-wise fine-tuned cosine as a feature)
- Fable review 2 (FABLE_PROMPT_2.md -> reply in chat): skip the France-blank probe (use D at France 0.90 vs 0.80: both valid files,
  dLB/0.1497 = France dF); decoy density can't explain the gap (FP_decoy budget only +0.0008); main lever = fine-tuned scorer.
  Agreed plan: run E = D + fold-wise fine-tuned e5 cosine. Not done (reasons): French normalizer changes (would need all 300M pairs'
  features rebuilt; unvalidatable), orphan kNN features (new S2/S3xS2/S3 search code), decoy-injection retrain.
- New code: s18_ftcos.py (train m0 on S1 buckets [0,2,4,6] / m1 on [1,3,5,7], 300k pairs each, s17 recipe; encode all records with
  both, tokenized once; ftc features for every union pair: ftc, ftc_rk, ftc_gap, ftc_rk_id, ftc_gap_id; train pair in fold f uses
  m{1-f}; test has _m0/_m1 columns -> s4_train's stacked test path gives booster g the model it was trained with).
  s17: training loop moved into finetune() (behaviour unchanged). s4_train: ftc_path added to extra_paths.
  s15_blend: also writes <name>_t80 (t 0.80 incl. France); `s15_blend.py write <name> <t> <t_fr> <stages>` = write-only.
  build_kaggle.py: KAGGLE_runE.ipynb (stage 9/10 -> v3s9, blend910 (+_fr90, _t80)), KAGGLE_variantD.ipynb (blend78ft_t80 from D's preds).
- Rehearsal (smoke_kaggle/logE.txt): exit 0, 6 files PASS. Fold models on the sample: m0 recall@20 +0.0028, m1 0.9947 -> 0.9963.
  ftc sanity (sample): true pairs mean ftc 0.847 vs 0.392, ftc_rk_id 1.003 vs 4.32; test m0 vs m1 corr 0.9991, means 0.446 / 0.451.
  stage 9: 89 features (D's 84 + 5). variantD: PASS; t80 == fr90 on the sample (sample has no France rows).
- run E preflight (s19_preflight.py) added at the top: opens all 97 inputs, checks emb sizes, 2 GPUs, e5/internet, disk. Tested locally: OK line, and FAILED lists (missing knnft/ce_v3, truncated emb, 1 GPU). Rest of run E identical to the rehearsed script (diff).

## 2026-09-27 05:37 UTC — Kaggle run D DONE (KAGGLE_runD.ipynb: run C union + s16 competition features, stage 7/8)
- feat3 train 162,712,539 pairs, recall 0.9937 (= C). idcomp on 162M pairs: train 165 s, test 134 s (no RAM problem).
- v3s7ft [7]: OOF best t=0.7 **0.98702** | at 0.80: 0.98685 | US 0.9876, India 0.9861
- stage 8 alone: t=0.7 0.98712
- blend78ft [7, 8]: OOF best t=0.7 **0.98722** | at 0.80: **0.98698** | US 0.9878, India 0.9864
  vs C blend56ft: 0.98657 / at 0.80 0.98593 -> **+0.00065 best, +0.00105 at 0.80** (competition features add on top of C, as in B).
- Matched pairs: v3s7ft 5,843,852 / _fr90 5,803,923; blend78ft 5,843,488 / _fr90 5,804,124. All 4 PASS.
- Gate for upload (OOF@0.80 > 0.98593): passed -> upload blend78ft_fr90, then KAGGLE_variantD -> blend78ft_t80 (France 0.80).
- LB: blend78ft_fr90 **0.98135** (C 0.979638: +0.0017 LB for +0.00105 OOF@0.80). blend78ft_t80 (France 0.80) 0.981161.
  France-threshold probe: dLB = -0.00019 -> France dF0.5 = -0.00019 / 0.1497 = **-0.0013** going 0.90 -> 0.80.
  => France probabilities are over-confident (pairs with p 0.80-0.90 in France are more often wrong than calibration says);
  0.90 stays (Fable's calibration argument did not hold for France). Higher France thresholds untested.
- Run E on Kaggle (started 04:16 UTC): PREFLIGHT OK (97 inputs, 2x T4, original e5 found in mlc26-v3/e5_small, disk 1119 GB).
  Fold models (held-out S1 of the other fold, 20k S1 / 300k db): m0 recall@20 0.9810 -> **0.9986** (+0.0176), m1 0.9815 -> 0.9986 (+0.0171).
  Encode both models: train 4743 s, test 4602 s (parallel). ftc features: train 162.7M pairs 610 s, test ~127.6M pairs 672 s. 06:12 UTC -> D steps.
- Built KAGGLE_variantE.ipynb (blend910 at t 0.80 / France 0.95 from E's preds; local test PASS).

## 2026-09-27 ~08:00 UTC — Fable review 3 -> run R (decode-time rules validated on labelled OOF)
- Fable 3 (FABLE3.TXT): France inference holds (France ~0.95-0.96); both last slots should go to France (0.983 ranks like 0.981).
  Rank: rules (a) validated on the US/India generic-name subset (drop a set only if its precision < 0.7); LLM veto (b) checkable on
  US/India; skip pseudo-labels (c) and new features (d). Keep France 0.90 (not 0.95 blind). Caution: E may add French matches.
- Problem: no run saved train OOF predictions (only in /tmp). Fix: run R rebuilds TRAIN features of D's union and recomputes OOF p7
  from D's saved stage-7 models; a reproduction check (must equal D's 0.98702) guards it.
- New: s20_rules.py (oof / rules / apply); rule families num (generic + no shared house number), noad (generic + record without
  address), cap (per-S1 match cap = train GT quantile), amb (record's top-2 S1 share the same generic core, p gap < d).
  A variant passes if OOF dF0.5 > 0 at t 0.80 and dropped precision < 0.70; best passing per family, then combined.
  Also saves state/oof_band.parquet (labelled OOF matches with p < 0.99, for an LLM-veto check).
  s19_preflight: BER_MIN_GPUS (default 2). Notebooks: KAGGLE_runR.ipynb, KAGGLE_applyE.ipynb (rules on E's preds, CPU).
- Rehearsal (smoke_kaggle/logR.txt): exit 0; OOF reproduction exact (0.99357 / at 0.80 0.99330 = D rehearsal); applyE PASS and
  byte-identical to E's blend910_fr90 when no rule fires. Sample too small (70 generic S1) for meaningful rule numbers.
- France diagnostic on D's blend78ft_fr90 (output_D/, scripts scratchpad fr_diag.py / fr_sample.py), vs train truth:
  empty S1 France 5.6% (train truth 5.6%, test US 5.7, India 5.8); matches/S1 France 3.31 (US 3.37, India 3.35, truth 3.46);
  matched S1 with both S2+S3: France 81.2% (US 83.4, India 82.8, truth 85.2); S1 above train max (n2>5 or n3>6): France 5.0 per 10k
  (US 0.4, India 0.6) = ~130 S1 only. name_eq France 27% of pairs (US 13.7, India 6.6) = generator style, not an error signal.
  Samples: French matches look right, incl. generic names (the house number decides between same-name S1). Magnets (11 matches)
  are genuine duplicates plus occasionally a DIFFERENT generic-word name at the same address (Nature Amis vs "Nature Parents SAS",
  "SAS Nature Maternelle") -> possible FPs, but only ~130 S1. => No visible France breakage; France ~0.95 remains an inference.
- Run R attempt 1 (Kaggle, 07:55-09:00 UTC): preflight OK; train rebuild recall 0.9937; **OOF reproduction EXACT: stage 7 best
  t=0.7 F=0.98702, US 0.98762 India 0.98611 (= run D)**. Then crashed in rules: Kaggle's polars lacks Expr.get(null_on_oob=).
  Fix: ambiguous() uses first/last + count; 'amb' via a joined column (no Series.implode); OOF p>=0.5 saved to
  state/oof_p7_ge05.parquet (a rerun with R's Output as input skips the 65-min rebuild); `s20_rules.py selftest` (synthetic
  example, all 4 rule families, checked by hand) runs first in run R and applyE. Local rehearsal: full path, skip path, applyE -> PASS.
- Run E stage 9 (Kaggle, 08:53 UTC): 89 features; top gain now ftc_rk_id / ftc / ftc_gap_id (the fine-tuned cosine dominates).
  v3s9 [9]: OOF best t=0.725 **0.98722** | at 0.80: **0.98708** | US 0.9877, India 0.9865. (D single v3s7ft: 0.98702 / 0.98685;
  D blend78ft: 0.98722 / 0.98698.) -> single-model gain +0.0002 at 0.80 only: the fine-tuned cosine mostly overlaps existing signal.
  Matched pairs v3s9 5,841,028 / _fr90 5,809,083 / _t80 5,819,409. 3 PASS. Stage 10 training.
- **Run E DONE 09:55 UTC.** stage 10 alone: best t=0.725 0.98739 (US 0.98784, India 0.98672).
  **blend910 [9, 10]: OOF best t=0.725 0.98744 | at 0.80: 0.98726 | US 0.9879, India 0.9868** (D blend78ft: 0.98722 / 0.98698,
  US 0.9878, India 0.9864) -> +0.00028 at 0.80; both countries up -> upload gate passed.
  Matched pairs blend910 5,840,691 / _fr90 5,809,259 / _t80 5,819,533 (D blend78ft_fr90 5,804,124: E adds +5,135 = +0.09%).
  France pairs with p in [0.80, 0.90): E 10,274 (t80 - fr90).
- LB: blend910_fr90 **0.981678** (D 0.98135: +0.00033 LB for +0.00028 OOF@0.80). CV->LB gap ~ -0.0056 unchanged since C.
- LB board (user, ~10:15 UTC): 36 teams at 0.990, next 40 at 0.989 (cut-off ~rank 76). Our US/India OOF 0.9873 is itself below it -> scorer is weaker everywhere, not only France.
- Fable review 4: no 5-h path to 0.989 with > ~15% odds. Top teams: fine-tuned cross-encoder as main scorer + metric-aware
  decoding. Plan: per-S1 expected-F0.5 decoder on saved p (validated on OOF), France p^2, rules if they pass; CE stretch skipped
  (friend's account can't see our outputs; +0.001 estimate). NOTE: we measured expected-F decoding on day 2 (s9, v2 stage 1):
  -0.00003 OOF -> Fable's +0.002-0.005 is doubtful; re-measure on run R's OOF before use.
- New: s21_efdecode.py (selftest / tune <oof> / apply; France own floor+gamma; q = exp(gamma*log p), no pow(expr)),
  KAGGLE_decodeE.ipynb (CPU; params EF in build_kaggle.py, rebuilt after tuning). Local: selftest OK, apply PASS on sample.
  Error-budget script ready (scratchpad err_budget.py).

## 2026-09-27 ~11:00 UTC — run R results + local analysis of run D's OOF (output_r/)
- Run R: SELFTEST OK on Kaggle polars 1.35.2; OOF reproduction exact again (0.98702). Rules at t 0.80 (base 0.98685):
  num 0.97/0.99/1.01: dropped 13.6k/29.5k/161k pairs, precision 0.935/0.962/0.992 -> dF -0.0005/-0.0014/-0.0102 (FAIL);
  noad: 6.5k, prec 0.736, dF -0.00002 (FAIL); cap: prec 0.95-0.98 (FAIL); amb 0.1: 1,351 dropped, prec 0.546, dF +0.00003 (PASS).
  => Fable's generic-name rules remove mostly TRUE matches: the model already handles those cases.
- Expected-F decoder (s21) on full OOF: best floor 0.7 gamma 1.0 = 0.98706 vs global t 0.80 0.98685 (+0.00021) but vs best
  global t 0.70 0.98702 only +0.00004 -> no real gain (same as day 2). Not used (would loosen France/US/India vs the LB-proven 0.80).
- Error budget (D stage 7 OOF, t 0.80, F 0.98685; lost F 0.01315 total):
  FN p<0.5 or not a candidate 200,172 pairs (+0.0084 if fixed) | FN p 0.5-0.8 60,619 (+0.0026) | FN stolen 7,669 (+0.0012) |
  FP_owned 7,179 (+0.0007) | FP_decoy 9,196 (+0.0008) | FP_single 1,119 (+0.0005). Lost F by S1: missed 1 / extra 0 = 0.0091,
  extra 1 = 0.0018, missed 2 = 0.0017. Samples: most FN (all buckets) are S2/S3 records with an EMPTY address and a name shared
  with other S1 (the unresolvable class of the 0.9939 oracle) or pseudonym names / website names with the same address.
- Last sprint: s22_stack.py = set-level XGB second stage on single-model p (>= 0.5) + record hashes (S1 #confident / per source,
  record #S1 candidates, rival S1 p / #confident, sibling same address / name, no-address, S1 name frequency). 2-fold by S1 bucket.
  Laptop experiment: t 0.80 0.98685 -> 0.98763. **Exact s22 code on full train OOF (records rebuilt from raw TSVs):
  t 0.75 0.98698 -> 0.98777 | t 0.80 0.98685 -> 0.98757 (+0.00072) | t 0.85 0.98651 -> 0.98741.** Top gain: r_n, r_noaddr,
  s1_name_n, s1_nconf. Test: D's p7 per model (p_m0 / p_m1) stacked separately, mean; decode t 0.80 / France 0.90.
  KAGGLE_stack.ipynb (CPU; preflight + selftest; asserts OOF gain before writing). Local rehearsal exit 0 + PASS.
  Estimate LB ~0.982 (D 0.98135 + 0.0007); E blend910 OOF@0.80 was 0.98726, stack7 0.98757.
- Perfect-judge ceiling on the uncertain band (OOF p in [0.5, 0.995): 752,428 pairs, 92.4% true): 0.98685 -> 0.99138 (+0.0045).
  => even a flawless re-scorer on US/India-type errors caps at +0.0045 OOF.
- Run X built (user: high-risk last push): s23_ceband.py = s8_ce (mmarco-mMiniLMv2 CE, fold-wise) retargeted to the uncertain band of
  the final model (train on labelled OOF band, score OOF band with the other fold's model + test band with both); s22 stacker with
  BER_STACK_CE=1 adds the CE logit. KAGGLE_runX.ipynb (GPU T4 x2, Internet On): writes stack7 and stack7ce, checkpoints CE models +
  scores to state/. Local rehearsal exit 0, both files PASS.
- 12:30 UTC: Kaggle GPU queues full (last-day rush). Final: run X if it starts, else KAGGLE_stack (CPU) or applyE. E remains the latest upload (0.981678).
- KAGGLE_stack (CPU, 12:53 UTC): SELFTEST OK (polars 1.35.2); OOF at t 0.80 base 0.98685 -> stacked **0.98762 (+0.00077)**;
  US 0.98750 -> 0.98819, India 0.98587 -> 0.98676. Test matches: France 859,006 -> 863,143 (+0.48%), India +0.22%, US +0.21%;
  total 5,803,923 -> 5,819,054. PASS. -> final upload candidate stack7.
- **Run X DONE 13:54 UTC.** CE band scoring: OOF 752,428 pairs (~2,220 pairs/s, 1 model), test ~748k pairs (~1,180 pairs/s, 2 models).
  stack7 (no CE): OOF 0.98760 (+0.00075; US 0.98816, India 0.98676); France 3.328/S1.
  **stack7ce (+ CE logit): OOF 0.98835 (+0.00150 over plain p7 at t 0.80); US 0.98851, India 0.98811 (+0.0022)**; France 3.328/S1
  (= stack7, no French jump); matched pairs 5,817,675. PASS. -> FINAL upload: stack7ce.
- **FINAL LB: stack7ce = 0.983021** (rank ~640). E 0.981678 -> +0.00134; predicted from OOF (D 0.98135 + 0.00137) = 0.98272 ->
  transfer held (~1:1, slightly better). Day 3 total: 0.97735 -> 0.983021. Cut-off for consideration was ~0.989.
