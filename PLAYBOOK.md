# Amazon ML Challenge — Playbook

## Rules (non-negotiable)
1. **No wasted compute.** Anything that takes >5 min goes through `cached()` / `shard_map()` / `Ckpt` (kit.py). Checkpoint by **time** (every ~20 min) not by step. Keep only the last 2.
2. **Cheap re-runs.** Pipeline = separate stages, each saved to disk: `raw → clean → features/embeddings → model → OOF+test preds → submission`. Changing stage N never recomputes stages < N.
3. **No hallucination.** Every claim about the data comes from code output pasted into LOG.md. Every score in LOG.md was actually measured. "I think" goes under *Hypotheses*, never under *Results*.
4. **No OOM / disk-full.** Check `mem()` before and after every big step. Rules below.
5. **Smoke first.** Every new script runs on 1% of data end-to-end (incl. writing a submission file) before the full run. Costs 2 min, saves hours.
6. **Everything logged** in LOG.md: what ran, where, config, CV score, LB score, time, cost.

## Platforms
| | Colab free (you + friends) | SageMaker ml.g4dn.xlarge |
|---|---|---|
| GPU | T4 16GB (when quota allows) | T4 16GB |
| RAM | ~12GB | 16GB |
| Cost | free, but GPU quota runs out, sessions get cut | paid per hour while running (check exact $/hr on the AWS pricing page for your region) |
| Use for | EDA, smoke runs, parallel experiments across accounts | long runs that must not be interrupted |

**Local laptop** (RTX 3050 6GB, 16GB RAM, 12 CPU threads): free, no quota, never disconnects. Use `D:\Amazzon\.venv` (CUDA torch). Best for: EDA, LightGBM/sklearn, smoke runs, small-model inference/embeddings, LoRA on small models. 6GB VRAM → anything >~1B params goes to Colab/SageMaker. Ampere → bf16 works here (unlike T4).

**Budget:** $140 AWS credit. GPU quota requested (pending). Spend only on runs that already passed a smoke test locally/Colab.

- **Colab:** data on local `/content` disk (fast), checkpoints/cache on Drive (persistent, 15GB free). Drive is slow for many small files → copy data zip to `/content` and unzip there.
- **SageMaker:** the "5 GB" is the storage volume — too small for images/models. Increase the space's storage *before* starting. **Stop the space when not training** — it bills while running even if idle. Set idle auto-shutdown.
- **Moving things between accounts:** Google Drive shared folder or a private Hugging Face repo (`huggingface_hub.upload_folder`). Never re-compute something a friend already computed.

## Do RIGHT NOW (before 00:00)
- [ ] **AWS: Service Quotas → SageMaker → request ml.g4dn.xlarge** for the usage type you'll use (Studio JupyterLab / notebook instance / training job). New accounts often have 0 GPU quota and approval can take hours. Do this first.
- [ ] **AWS Budgets: alert at $20, $50, $80.** (Also reportedly one of the tasks that earns the extra free-plan credits — verify in the console.)
- [ ] Start a g4dn space once, run `nvidia-smi`, `python kit.py`, then **stop it**. Confirms quota + billing + storage works before it matters.
- [ ] Colab on every account: mount Drive, `nvidia-smi`, upload kit.py to `MyDrive/amzml/`, run `python kit.py`.
- [ ] Pre-download likely model weights to Drive / HF cache so you don't burn GPU time downloading (decide once problem type is known; a text encoder + a small vision model + LightGBM covers most cases).
- [ ] Assign roles with friends: who owns which account, who runs what. One shared LOG.md (Google Doc or this file in a shared Drive).
- [ ] Read last year's winning write-ups (search "Amazon ML Challenge 2024/2025 solution github"). Past editions I recall — **verify, don't trust**: 2023 product length prediction (text), 2024 extracting entity values from product images, 2025 product price prediction (text + image, SMAPE). Past rules have restricted external APIs and model size — read this year's rules carefully.
- [ ] Sleep if you can. 72h is a marathon.

## Hour 0 protocol (dataset drops)
1. Read problem statement + **metric** + **rules** (allowed models/licences, external data, submission format). Write them in LOG.md.
2. Download data once → persistent storage. Record sizes, row counts, columns.
3. EDA on CPU (no GPU needed): nulls, target distribution, train/test differences.
4. Implement the **exact metric** locally + a CV split that mirrors train/test.
5. Dumbest baseline (median / TF-IDF + linear / LightGBM) → `check_sub()` → **submit #1 within the first few hours**. Validates format + CV↔LB correlation.
6. Only then GPU work.

## Submission budget (5/day × 3 = 15)
- Day 1: baseline + 2–3 real experiments. Keep 1 spare.
- Day 2: best single models. Only submit when local CV improves meaningfully.
- Day 3: keep **2 for the end** — final ensemble + safety.
- Always log CV next to LB. If they disagree, fix the CV before trusting either.
- Save OOF + test predictions for every model (`cached`) → ensembling is free, zero GPU.

## Fine-tuning without crashes
- **LoRA/QLoRA** over full fine-tune: small checkpoints (MBs), fits T4. `Ckpt` saves only trainable params.
- T4 has **no bf16** → use fp16 (`fp16=True`), not bf16.
- Memory knobs, in order: smaller batch + `gradient_accumulation_steps`, `gradient_checkpointing=True`, shorter max_len, 8-bit optimizer, 4-bit base model.
- Find max batch size on a 200-row smoke run, then use ~80% of it.
- Data: read with explicit dtypes, `float32`/`float16` not `float64`, `del df; gc.collect()` after use, `np.load(mmap_mode="r")` for big arrays.
- Inference: always `shard_map()` → a crash at 90% keeps 90%.
- Images: download with `shard_map` (skip existing), resize once, cache.

### HF Trainer checkpointing
```python
from transformers.trainer_utils import get_last_checkpoint
out = str(ROOT / "ckpt" / "run07")
args = TrainingArguments(out, save_strategy="steps", save_steps=500,  # tune to ~20 min
                         save_total_limit=2, fp16=True, ...)
trainer.train(resume_from_checkpoint=get_last_checkpoint(out) if os.path.isdir(out) else None)
```
