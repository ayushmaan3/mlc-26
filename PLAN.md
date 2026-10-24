# Plan — Business Entity Resolution

## Measured facts (train, from streaming EDA on 2026-09-25)
| | S1 | S2 | S3 |
|---|---|---|---|
| train rows | 2,206,821 | 5,034,616 | 5,285,603 |
| test rows | 1,732,544 | 4,887,273 | 5,082,316 |
| train rows matched to an S1 | — | 3,693,619 (73%) | 3,944,746 (75%) |

- Singletons (S1 with 0 matches): **5.6%**. Matches per S1: mostly 2–5, max 10+ (571 rows).
- **Every matched S2/S3 id appears exactly once in the ground truth** → each S2/S3 record belongs to *at most one* S1. Decode as an assignment: each S2/S3 goes to its single best S1 or none. Big precision lever.
- ~25% of S2/S3 match nothing, and some look very similar to real entities ("Holloway Peak Inc Seafood") → hard negatives exist.
- Empty address: ~3.4% of S2/S3; S1 never empty.
- Countries: train US/India. Test S1: US 663k, India 810k, **France 259k (15%, unseen)**.
- Noise seen: names in Devanagari/Bengali/Telugu/Kannada script (transliterated English words, e.g. "सुप्रीम प्रोडक्ट्स प्राइवेट लिमिटेड" = Supreme Products Private Limited), state names in native script, website/handle-style names (`diamondlogisticslabs.com`, `@willoughbycertified`), junk tails (`(ID: 76158)`, `#45992`, phone numbers), typos/leetspeak (`D0soft`), accents (`Ínc`), token shuffles, NULL tokens, house-number ranges (`156-158`), component reordering, **some matches with a completely different name and the same address** ("Irinyla", "Miraveraio").
- Address (house number + street + city) looks like the strongest signal; name second.

## Rules that constrain us
- Final model: **MIT/Apache-2.0 licensed, ≤ 8B params.** LightGBM (MIT) is fine. Any embedding model: check licence first.
- **No external APIs/lookups** (geocoding, business registries, internet data). → No hosted LLM APIs.
- `country` = open set. **No country one-hot / hard-coded {US, India}.** Features must be language-agnostic so France works.
- Output: `matching_results.tsv` + `candidate_pairs.tsv`. Matches ⊆ candidates. Run `utils/validate_submission.py` before every upload.
- Zip with runnable `src/`, README, pinned requirements, filled `Documentation_template.md`. Top teams get audited → keep code reproducible from day 1.

## Metric logic
Macro F0.5 per S1, singletons included. A wrong match on a singleton = 0. On a normal S1, adding one wrong id hurts more than missing one right id. → Prefer precise matches, tune the threshold for **macro F0.5 on validation**, never a fixed 0.5.

## Pipeline (each stage cached to parquet; changing stage N never re-runs < N)
0. **Load → parquet** (polars). TSV read once, ever.
1. **Normalize**: lowercase, strip accents, Indic → Latin transliteration, drop junk (ids, phones, `#123`, `NULL`), canonicalize legal suffixes (pvt/private, ltd/limited, llc, inc, corp, l.l.p.), street words (st/street, rd/road, ln/lane, ave, dr, ct, blvd), US state names ↔ codes, native-script state names. Extract: house-number tokens, street tokens, city tokens, PIN/ZIP, website/handle → name tokens.
2. **Blocking** (union of keys, within country, cap huge blocks):
   - house number + first street token
   - normalized name (exact) and rarest 2 name tokens
   - ZIP/PIN + name token
   - later: char-n-gram TF-IDF / embedding nearest neighbours for the leftovers.
   Measure on train: **recall ceiling** and **candidates per S1**. Target ≥ 97% recall at ≤ ~30 candidates/S1.
3. **Features** (rapidfuzz, vectorized): name ratio / token_set / partial, address ratios, house-number match, street/city/ZIP overlap, IDF-weighted token overlap, missing-field flags, length diffs, and **competition features**: this pair's score vs. the best score for the same S2/S3 across all S1s and vs. the best for the same S1.
4. **Model**: LightGBM binary, split **by S1 entity** (group split). Train on full-train candidates so negatives have real density.
5. **Decode**: each S2/S3 → argmax S1 if prob ≥ t; tune t (plus maybe a relative-to-top rule) for macro F0.5 on validation.
6. **Write** both TSVs → validator → upload.

## Validation
Blocking + features on **full train** (same scale as test). Hold out 20% of S1 entities (with their candidates) as validation. Implement the exact macro F0.5. Log CV next to LB for every submission.

## Compute
This is mostly **CPU + RAM** work, not GPU. GPU only matters later (optional embedding model for names in other scripts / ANN blocking).
- **RAM is the real limit:** the laptop had only 2.0 GB free during EDA. Close apps before running. Use polars (lazy / streaming), float32, categorical country, int64 ids (strip the `S2-` prefix).
- Laptop: dev + small samples. Colab CPU/High-RAM or SageMaker for full train+test runs.

## Timeline
- **Day 1:** hour 1: rule baseline → submission #1. Then normalization + multi-key blocking + recall measurement → LightGBM v1 → submission #2–3.
- **Day 2:** transliteration quality, better blocking recall, feature iterations, error analysis on false positives → submissions only when CV improves.
- **Day 3:** final tuning, (optional) embedding features, reproducibility run from scratch, docs + zip. Keep 2 submissions for the end.

## First submission (~1 hour target)
Rule baseline, no ML: block on (country, house number, first street token) ∪ (country, normalized name). Score = weighted rapidfuzz name + address similarity. Assignment decode. Pick the threshold on a train sample for macro F0.5. Validator → upload. Purpose: prove the full pipeline and format end-to-end, and get a first CV↔LB data point.
