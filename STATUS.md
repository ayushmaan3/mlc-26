# Team status: Amazon ML Challenge (entity resolution)
Updated 2026-09-25 18:20 UTC. Deadline 2026-09-28 00:00 IST. Metric: macro F0.5 per Source-1 entity.

## Scoreboard
| # | submission | our validation (OOF) | leaderboard |
|---|---|---|---|
| 1 | v1: LightGBM, 1/8 of train, blocking recall 0.933 | 0.9159 | 0.906 |
| 2 | v2 stage 1: transliteration + phonetic keys, XGBoost on all train | 0.9634 | **0.952** |
| 3 | v2 stage 1 + per-entity expected-F decoding | 0.9634 | **0.953** (tie; noise) |
- Validation runs about 0.011 above the leaderboard, consistently (v1 and v2). For LB 0.98 we need OOF ≈ 0.991.
- Rank ~300; many teams at 0.969–0.971.

## Key measured findings
- **Blocking recall is the ceiling.** Hand-built keys (v2) reach 0.950 of the true pairs.
  **Embedding kNN (multilingual-e5-small, top-20) + keys: 0.9784**, for +12.8 candidate pairs per entity. That's the next big step.
- India is the weakest country (native scripts, landmark-style addresses). France (test only) doesn't look worse by model confidence.
- The hard pairs: same street and city, house number off by 1–2, generic name words or legal forms swapped (SARL/EURL), acronyms.
- A global threshold is already near-optimal; per-entity decoding gave +0.00 on OOF.

## Where things are (Google Drive `MLC'26`)
| folder | contents |
|---|---|
| `work/` | base data: records, ground truth |
| `work_v2/` | v2 pipeline state: normalized tables, candidates, features (16), stage-1 models + predictions, competition features |
| `work_emb/` | train embeddings (9.6 GB), kNN candidates for train (8), recall report |
| `code_v2/`, `code_v3/` | pipeline code (v3 = latest) |
| `models/` | e5-small, e5-base, MiniLM-L12, mmarco cross-encoder (all MIT/Apache, offline) |
| `wheels/` | pip wheels for offline install |
| `output_s1/`, `output_s1_ef/` | submissions 2 and 3 |
| `output_v2/`, `output_s3/` | stage 2 / stage 3 submissions (being produced overnight) |

Local: `D:\Amazzon` holds the same code (`code_v2/`), `MAIN_pipeline.ipynb`, `PARALLEL_embeddings.ipynb`, `LOG.md` (every measured number), `PLAN_v3.md`, `RESUME.md`.

## Running now
- Account 4, `MAIN_pipeline.ipynb` cells 1–4: stage 2 (~1–1.5 h) → `output_v2`, then cross-encoder (~2.5 h) → stage 3 (~1 h) → `output_s3`.
  Everything finished is saved to Drive as it goes; if the session dies, re-run cells 1–4 on any account and it resumes.

## Ready to run (verified locally end to end)
- `PARALLEL_embeddings.ipynb` (test side only, ~2.5 h) -> `work_emb/test_emb.f16` + `knn_test_*`
- `V3_pipeline.ipynb` (needs the above) -> `output_v3s1`, `output_v3s2`, `output_v3s3` (~7 h over 2 sessions, resumable)

## Next (in order)
1. Morning: read cell 5 (stage 2 and stage 3 OOF), submit B (`output_v2`), then C (`output_s3`). Only upload files the validator marked PASS.
2. **Blocking v3** (the big lever): encode test + kNN test (~2.5 h, parallel account), features for the new pairs, retrain stages 1→2→3.
   Target OOF ≥ 0.985.
3. Final day: decode tuning, keep 2 submissions in reserve, write `Documentation_template.md`, build the code zip.

## Rules we follow
No external data or APIs. Every model is MIT/Apache-2.0 and ≤ 8B parameters (largest used: 278M). `country` is treated as an open set.
