# Plan v3: from LB 0.906 to ≥ 0.972

Deadline: 2026-09-28 00:00 IST (72 h from the start). Submissions: 5/day.
Current: LB 0.906 (exp03, CV 0.9159), rank ~180. The target needs +0.066 on the leaderboard.
Measured CV→LB gap: −0.010. Plan for it: aim for **OOF ≥ 0.980**.

## STATUS (updated 2026-09-25 18:20 UTC)
| phase | status | measured result |
|---|---|---|
| 0 v2 stage 1 | done | OOF 0.96339 (t=0.65), LB 0.952 (gap -0.011) |
| 0 v2 stage 1 + expected-F decode | done | OOF 0.96336, LB 0.953 (tie, noise) |
| 0 v2 stage 2 | running overnight (account 4, MAIN_pipeline.ipynb) | - |
| 4 cross-encoder + stage 3 | queued after stage 2 (same run) | - |
| 2 embedding kNN blocking | train side measured | recall 0.9503 -> **0.9784** (kNN@20, +12.8 pairs/S1) |
| 1 error budget / 3 features v3 / 5 decode | decode tried (no OOF gain); rest not started | - |

Phase 2 is confirmed as the biggest lever. Next session = "blocking v3" (below).

## Blocking v3 (next, replaces the old phase-2 text)
1. Test side: encode test (~2.3 h, CPU-bound tokenizer) + kNN test (~20 min). Train side is already on Drive.
2. New pairs = kNN@20 minus cand2, per bucket (~28M train, ~22M test). Features for new pairs only (same s3 code; nk* = 0),
   plus new features for ALL pairs: emb_cos, erk (kNN rank, NaN if not in kNN), in_knn, in_keys.
   Cosine computed from RAM (load only the rows each bucket needs, in id order), not by random memmap reads.
3. feat3 = feat2 + new pairs. Retrain stage 1 -> comp -> stage 2 -> (cross-encoder on the new uncertain pairs) -> stage 3. Tune on OOF; submit.
4. Budget: 5-6 h GPU runtime (fits one account). Target OOF >= 0.985 (LB >= 0.975 at the observed -0.011 gap).

## Where the points are (reasoning from measured numbers)
- Macro F0.5 per S1. 5.6% of S1 are singletons (1.0 only if we predict nothing).
- The pair recall ceiling after blocking is **0.950** (v2, measured). Uncapped, it is 0.963, so the top-60 cut loses 1.3 pts and missing keys lose 3.7 pts.
  With ~5% of true pairs unreachable, F0.5 can't reach 0.97 however good the classifier is.
  → **Blocking recall is lever #1.** Target ≥ 0.985 at ≤ 60 candidates per S1.
- In exp03, about the same number of true pairs sat in the candidate set but scored below the threshold (122.9k on val) as were never generated (127.7k).
  → **Classifier recall at high precision is lever #2** (better features plus a cross-encoder).
- FP 74.6k on val, 32k of them caused by the val-bucket decode. The global OOF decode (v2) fixes those.
  → **Competition/assignment is lever #3** (stage 2 plus decode rules).
- India 0.885 vs US 0.937: most of the loss is India (native scripts, landmark-style addresses). France is unseen (15% of test S1).

## Phase 0: finish v2 (about 3 h, already coded and saved)
1. `RESUME.md` cells 1–3, then status cell 4.
2. `run_rest.sh` does: test features 3–7 → stage 1 → **writes `MLC'26/output_s1`** (submit it: LB #2) → comp → stage 2 → `output_v2` (submit: LB #3).
3. Log OOF F0.5 overall / by country and the LB numbers in LOG.md.

## Phase 1: error budget (30 min, CPU, cheap) — decides what to build next
Script `s7_errors.py` on the stage-2 OOF at the tuned t. Split 1 − F into:
blocking miss / in candidates but below t / stolen by another S1 / FP on a singleton / FP on a matched S1,
each by country × source (S2 vs S3) × empty-address × non-Latin. Print 20 examples per bucket.
Rule: work on the biggest bucket first. The phases below are ordered by the expected size of each bucket, not fixed.

## Phase 2: blocking v3 — embedding kNN (target recall ≥ 0.985)
- Encode every record with **multilingual-e5-small** (MIT, verified 6.1k names/s on T4 fp16 while the CPU was busy; probably ~10k/s idle).
  Two texts per record: `query: <name>` and `query: <name> | <address>`. About 22M strings × 2, so ~1–1.5 h per split. Store float16 on disk (≈ 17 GB/split for 384-d; OK on disk, never fully in RAM).
- kNN on GPU with plain torch matmul in blocks, **within country** (no faiss needed):
  India test ≈ 0.8M S1 × 5M S2/S3 × 384-d ≈ 1.5e15 FLOP ≈ 1–2 min on a T4 plus top-k. Keep top-20 per S1 for each text type.
- Union with the v2 keys. Replace the hand-weighted `sc` rank with a **learned cheap ranker** (LightGBM on nk0–8 + the two cosines + rank-in-kNN, trained on the uncapped bucket 0), and keep the top-60 by that score.
- Measure: recall, pairs/S1 (keep ≤ 60), then recompute features only for the new pairs.
- Estimate (not measured): recall 0.950 → 0.975–0.985.

## Phase 3: features v3 (cheap, big for precision)
- Embedding cosines: name, name+addr (e5-small), plus rank of the cosine within the S1 and within the id.
- **Cross-source consistency:** for pair (s1, x), the best similarity of x to the other high-p candidates of that s1 from the *other* source.
  True clusters agree across S2 and S3; false merges usually don't.
- Address structure: PIN/ZIP (6-digit / 5-digit) equal or missing, city token equal, house-number range contains (`156-158` ∋ `156`), unit/flat numbers.
- Name: acronym match (`ssprivate` ↔ Stuti Solutions Pvt), rare generic-suffix substitution (Center/Services/Trading), token-order-free prefix match.
- France (unseen): add French legal forms (sarl, sas, sasu, eurl, sa, sci — done), street words (rue, av, bd, chemin, impasse, allée — done), `cedex`, 5-digit postcode,
  `bis/ter`. Sanity check: predicted match rate and average p for FR test vs US/India (it can't be scored, only compared).

## Phase 4: cross-encoder re-ranker on uncertain pairs (the big model lever)
- Base: **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** (Apache-2.0, 118M, multilingual, already a pair classifier).
  Fallback: xlm-roberta-base (MIT, 279M) or e5-base (MIT, 278M).
- Input: `"<name1> | <addr1>" [SEP] "<name2> | <addr2>"`, max_len 128. Fine-tune 1 epoch on ~3–5M train pairs with stage-1 OOF p in [0.02, 0.98] (hard pairs only),
  split by S1 bucket exactly like the GBDT folds (2 models, OOF).
- Inference only on uncertain test pairs (p1 in [0.02, 0.98], probably ~10–20% of the 85M pairs). Estimate ~1.5–2k pairs/s on a T4 fp16 → 1–3 h. **This is the most GPU-heavy step; a SageMaker g5/A10G would halve it.**
- The CE probability becomes a stage-2 feature (not a replacement), so the GBDT keeps the structured signals.
- Estimate (not measured): +0.01–0.02 F0.5, mostly India and non-Latin.

## Phase 5: decoding (cheap, needs care)
- Global assignment (done). Add a relative rule: drop x from s1 if p < α · s1_max (tune α).
- Tune the threshold separately for S2 and S3 matches and per country (train countries only; France uses the global t).
- S1 cardinality: expected number of matches from Σp; don't exceed round(Σp)+1 per S1.
- Singleton guard: an S1 whose best p < t_single gets an empty list (tune t_single separately).
- All tuned on full-train OOF, never on LB.

## Phase 6 (only if time is left): transductive tricks
- Pseudo-labels: add very confident test pairs (p > 0.99 and a clear winner) as training rows. Uses only the provided test data, no external lookup,
  but check it reads as fair play in the documentation before using it.

## Compute plan
- Colab T4 (12.7 GB RAM, 2 vCPU): OK for GPU steps (embeddings, CE, XGBoost). CPU feature stages take ~8 min/bucket and memory is tight.
- **Recommended: SageMaker `ml.g4dn.4xlarge`** (T4, 16 vCPU, 64 GB, ≈ $1.5/h) or `ml.g5.2xlarge` (A10G, 8 vCPU, 32 GB, ≈ $1.5/h).
  $140 ≈ 90 h. 16 vCPU makes the rapidfuzz features ~8× faster, and 64 GB removes the OOM work. Data: Drive → laptop → `aws s3 sync` → instance.
- Laptop (12 threads, but only ~2–3 GB RAM free) is only for editing and small tests.

## Submission schedule (5/day)
- Day 2: s1 (v2 stage 1), s2 (v2 stage 2), then v3 blocking + features.
- Day 3: + cross-encoder, then decode-tuned final. Keep 2 submissions for the last hours.
- Always: validator PASS first; log CV and LB side by side.

## Prepared offline (verified licences; all on Drive `MLC'26/models` and local `D:\Amazzon\models`)
| model | licence | params | role |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 117.7M | embeddings for kNN blocking + cosine features |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 117.7M | second embedding view (weak across scripts) |
| intfloat/multilingual-e5-base | MIT | 278.0M | stronger embeddings / CE fallback base |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Apache-2.0 | 117.6M | cross-encoder re-ranker base |
Pip wheels (rapidfuzz, anyascii, faiss-cpu): Drive `MLC'26/wheels` (`pip install --no-index -f <dir> ...`).
Not needed: Qwen2.5 0.5B/1.5B (Apache-2.0, but far too slow for millions of pairs).
