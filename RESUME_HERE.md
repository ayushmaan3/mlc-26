# FINAL STATE 2026-09-27 ~12:30 UTC
- Best LB: E blend910_fr90 = 0.981678 (latest upload). Cut-off for consideration ~0.989 (76 teams). 1 slot left.
- Final-slot options (all validated on labelled OOF at t 0.80): KAGGLE_stack.ipynb (CPU) -> stack7, OOF 0.98757 (best);
  KAGGLE_runX.ipynb (GPU, queued) -> stack7 / stack7ce (cross-encoder gamble); KAGGLE_applyE (CPU) -> E + amb rule.
- Kaggle GPUs queued everywhere at the end; user: run X if it starts, else a CPU file.
- Laptop fallback prepared: scratchpad/fullw has train/test records + OOF; needs run D's p7_v3_test_*.parquet to run s22 locally.

# UPDATE 2026-09-27 ~04:15 UTC
- LB: blend56ft_fr90 = 0.979638 (best). Top 21 teams 0.990; need ~0.989. 4 submissions left (last day).
- Run D running (user account). Run E BUILT + rehearsed: KAGGLE_runE.ipynb (fold-wise fine-tuned e5 cosine features, stage 9/10).
  Inputs: mlc26-core, mlc26-emb, mlc26-v3 + Output of KAGGLE_runC; GPU T4 x2; Internet On (downloads original e5 if absent).
- KAGGLE_variantD.ipynb: after D, makes blend78ft_t80 (France 0.80) from D's preds (CPU, 5 min). Inputs: mlc26-core + D's Output.
- Upload plan: 1) blend78ft_fr90 if D OOF@0.80 > 0.98593  2) blend78ft_t80 (France dF = dLB/0.1497)  3) E blend910 at the better
  France setting if E OOF@0.80 beats D on US and India  4) best file last (check portal for final-selection control).

# UPDATE 2026-09-27 ~02:00 UTC
- Run C DONE: recall 0.9784 -> 0.9937; blend56ft OOF 0.98657 (A 0.98454). Upload **blend56ft_fr90** first (from KAGGLE_runC Output).
- Run B DONE: v3s7 0.98483 / blend78 0.98490 OOF (+0.0006 over A at t 0.80). Competition features help.
- Run D BUILT + rehearsed locally (smoke_kaggle/logD.txt, exit 0, 4 PASS): KAGGLE_runD.ipynb = C's union (knnft_* from
  KAGGLE_runC Output added as input) + stage 7/8. Outputs v3s7ft(_fr90) ~3 h, blend78ft(_fr90) ~3.7 h.
  Upload blend78ft_fr90 if its OOF > 0.98657 (C). Risk: s16_idcomp RAM on 162M pairs (B did 139M fine) -> set -e kills the run.
- Deadline 18:30 UTC.

# RESUME HERE (written 2026-09-27 ~02:30 IST / 2026-09-26 ~21:00 UTC, while the user sleeps)
New session: read this first, then TRACKER.md (submissions + findings) and LOG.md (every measured number).
Deadline: **2026-09-28 00:00 IST = 2026-09-27 18:30 UTC.** 5 submissions available on Sep 27 (IST day). None used yet at writing.
Metric: macro F0.5 per S1. Best public LB so far: **0.977** (output_v3s5_fr90, TRACKER #8). Target: ~0.988. Top: 0.9907.

## 1. What is running right now (Kaggle, account ayushmaan03, background "Save & Run All")
| run | notebook | status | outputs (Output tab of the saved version) |
|---|---|---|---|
| A | KAGGLE_run.ipynb | **DONE** 20:06 UTC | v3s5full(_fr90), blend56(_fr90), candidates/, state/ |
| B | KAGGLE_runB.ipynb | running (started ~19:50 UTC, ~6 h) | v3s7(_fr90) then blend78(_fr90) |
| C | KAGGLE_runC.ipynb | running (started after A, ~20:10-20:30 UTC, ~6-7 h) | v3s5ft(_fr90) then blend56ft(_fr90), state/knnft/ |
Inputs: datasets mlc26-core, mlc26-emb, mlc26-v3 (+ mlc26-ftmodel for C). Log of each run: `log_kaggle.txt` in its Output tab.

### Run A results (measured, full-data training; Colab stage 5 used half data)
- v3s5full [5]: OOF best t=0.675 **F0.5=0.98445** | at 0.80: 0.98410 | US 0.9869, India 0.9807
- stage 6 alone (511 leaves, seed 1, cols 0.7, mcw 40): 0.98440
- blend56 [5, 6]: OOF best t=0.675 **F0.5=0.98454** | at 0.80: 0.98419 | US 0.987, India 0.9809
- Full data vs half data: only +0.0004 OOF (0.9841 -> 0.98445). Blend adds +0.0001. => model capacity is NOT the bottleneck; candidates are.
- Matched pairs: v3s5full 5,769,339 / _fr90 5,725,482; blend56_fr90 5,725,325. Both validator PASS.
- blend56_fr90 = fallback file only (expected LB ~0.977-0.978, about the same as #8); not uploaded by user's choice.

### Run B = stage 7 (stage 5 + per-record competition features, s16_idcomp.py), same candidates as A
- Look for `v3s7 [7]: OOF best ...` and `blend78 [7, 8]: OOF best ...`. Compare with A's 0.98445 / 0.98454.
- Worth an upload only if clearly above A (>= ~0.9850).

### Run C = fine-tuned e5 retrieval (Fable idea #1) + full-data stage 5  <- the main hope
- Validation (FT_check.ipynb, s17_ft_retrieval.py): 100k held-out S1, db 1.17M, within-country kNN recall@20 0.9771 -> **0.9970**.
- Log checkpoints: (1) `fine-tuned e5 loads (transformers X): cos same-business 0.9xx, different 0.xx` ~10 min in;
  (2) ft_enc_*.log / ft_knn_*.log in the Output tab; (3) **`feat3 train: ... recall 0.9xxx`** must be clearly above A's **0.9784**;
  (4) `v3s5ft [5]: OOF best ...` (~5 h) and `blend56ft [5, 6]: OOF best ...` (~6.5 h).
- The fine-tuned model only ADDS candidates; all features still use the original e5 (FT saw 4% of train positives -> leak risk).
  CE feature is null for the new pairs. addr4 rebuilt for the new union.
- If it fails early: read the error in log_kaggle.txt. Model-load fallback is built in (original e5 config/tokenizer + FT weights).
  If knnft files were produced (state/knnft/), a rerun can add that Output as input and skip encoding (script links knnft_* from inputs).

## 2. Submission plan for Sep 27 (5 slots; keep >= 1 in reserve)
1. User decision: do NOT spend a slot on A blend56_fr90 now (OOF only +0.0005 over the 0.977 file). Use it only if B and C both
   fail, or with a slot left unused near the deadline.
2. When C gives v3s5ft: if its OOF > A's, upload **v3s5ft_fr90** (then blend56ft_fr90 if higher still).
3. B: upload v3s7_fr90 / blend78_fr90 only if OOF clearly beats A.
4. Always the `_fr90` variant (t 0.80, France 0.90): it has won on the LB every time (test has ~20% more decoys than train).
5. Final selection on Kaggle/portal: best public LB, tie-break by OOF. Then the final zip (section 4).

## 3. Possible next run if time allows (not built yet)
- "Run D" = run B's stage 7 on C's bigger union: add C's Output (state/knnft/) as input -> no re-encoding; ~4.5 h.
  Only if B shows stage 7 > stage 5 AND C shows gain. Build from build_kaggle.py (RUN_B + the knnft linking lines of FT_BLOCK;
  addr4 must be rebuilt as in RUN_C).
- Widen the CE band / run CE on the new C pairs: expensive (hours), skip unless lots of time.

## 4. Final deliverables (due with the last submission)
- Zip: output/ (matching_results.tsv + candidates/candidate_pairs.tsv of the chosen run) + code/business_entity_resolution
  (code_v2/*.py incl. s10_v3, s14_addr, s15_blend, s16_idcomp, s17_ft_retrieval, e1_emb, s4_train) + README + requirements
  + Documentation_template.md. Models used: multilingual-e5-small (MIT), fine-tuned copy, mmarco-mMiniLMv2-L12 CE (Apache), XGBoost.

## 5. Where things live
- Local: D:\Amazzon. Code: code_v2/. Notebook builders: build_kaggle.py (runs A/B/C), build_notebooks.py (Colab).
  Rehearsal: smoke_kaggle/ (run_C_local.sh, logC.txt = exit 0, 4 submissions PASS).
- Drive (main account, folder MLC'26): work/, work_v3/, models/, code_v2/. Spare Colab account Drive: MyDrive/e5_ft (fine-tuned model).
- Kaggle datasets (ayushmaan03): mlc26-core, mlc26-emb, mlc26-v3, mlc26-ftmodel.
- Fable/Luna reviews: FABLE.TXT, LUNA.TXT. Brief given to them: FRESH_EYES_BRIEF.md.

## 6. User working rules (keep)
Accuracy over submission size; refer to notebooks by FILE NAME (not "account 1/2"); never "Run all"; give simple exact cells;
never waste compute; no RAM/disk crashes; no hallucination - measured numbers only; keep TRACKER.md + LOG.md updated.
Kaggle: don't cancel a committed run (outputs are lost); at most 2 GPU sessions ran at once (C had to wait for A).

# FINAL ZIP (2026-09-28)
- SUBMISSION/TEAM_submission = organiser layout, final file stack7ce (md5 f2ca30b6...). Missing only output/candidate_pairs.tsv
  (download KAGGLE_runD Output candidates/candidate_pairs.tsv). Then: python SUBMISSION/finish_zip.py <team> (checks coverage, zips).
  Fill team name/members in Documentation_template.md.
