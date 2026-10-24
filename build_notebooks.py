"""Builds MAIN_pipeline.ipynb and PARALLEL_embeddings.ipynb from the verified code in code_v2/."""
import json
from pathlib import Path

C = Path(__file__).parent / "code_v2"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def writefile(path, name):
    return code(f"%%writefile {path}/{name}\n" + (C / name).read_text(encoding="utf-8"))


CODE_FILES = ["common.py", "s0_load.py", "s1_normalize.py", "s2_block.py", "s3_features.py", "s4_train.py",
              "s5_comp.py", "s6_write.py", "s8_ce.py", "s9_decode.py", "s10_v3.py", "s11_tune.py", "s12_expand_diag.py", "s13_errors.py", "s14_addr.py",
              "e1_emb.py"]


def nb(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


SETUP = r'''
# ---- Cell 1: config + mount + install (one Drive-auth click) ----
D = "/content/drive/MyDrive/MLC'26"      # shared folder, added to *this* account's My Drive as a shortcut named MLC'26
from google.colab import drive
drive.mount('/content/drive')
import os, subprocess, shutil
_free = shutil.disk_usage("/content/drive").free / 1e9
# Files count against the account that CREATES them (even inside the shared MLC'26), so THIS account's free space matters.
print(f"this account's free Drive space: {_free:.1f} GB  (V3 main needs ~6 GB, V3 helper ~7 GB; empty the Drive trash if it is lower)")
assert os.path.isdir(D), f"{D} not found: in Drive, open 'Shared with me' -> MLC'26 -> 'Add shortcut to My Drive'"
assert os.path.isdir(f"{D}/work_v2") and os.path.isdir(f"{D}/code_v2") and os.path.isdir(f"{D}/models"), "MLC'26 is missing work_v2 / code_v2 / models"
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    print(r.stdout[-6000:], r.stderr[-3000:])
sh(f'pip install -q --no-index -f "{D}/wheels" rapidfuzz anyascii 2>/dev/null || pip install -q rapidfuzz anyascii')
sh("python -c 'import rapidfuzz, anyascii, xgboost, torch; print(\"ok\", xgboost.__version__, torch.cuda.is_available())'")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2; df -h /content | tail -1")
'''

STATUS_MAIN = r'''
# ---- Status (re-run any time; cheap) ----
sh("date; echo '--- run_rest'; tail -12 /content/log_rest.txt; echo '--- run_ce'; tail -12 /content/log_ce.txt; "
   "echo; ls /content/work | grep -E 'feat2|xgb|val_stage|^p[123]_|comp_|^ce_' | tr '\n' ' '; echo; "
   "pgrep -af 'run_rest|run_ce|s3_features|s4_train|s5_comp|s6_write|s8_ce' | grep -v pgrep; free -m | head -2; "
   "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv")
for s in (1, 2, 3):
    p = f"/content/work/val_stage{s}.json"
    if os.path.exists(p):
        import json; d = json.load(open(p)); print(f"stage {s}: OOF F0.5 {d['f05']:.4f} at t={d['t']}  by country", {k: round(v, 4) for k, v in d['by_country'].items()})
'''

MAIN = [
    md('''
# MAIN pipeline: v2 stage 1 → stage 2 → cross-encoder → stage 3
**Runs unattended in the background.** The notebook only launches shell jobs and shows status, so a kernel crash can't take the job down.
Every step skips outputs that already exist, so after any crash or disconnect just re-run cells 1, 2, 3 and 5.

| step | what | ≈ time (T4) | output to submit |
|---|---|---|---|
| restore | copy data from Drive | 15 min | |
| run_rest | test features 3–7 → stage 1 | ~1.5 h | **`MLC'26/output_s1/matching_results.tsv`** (submission A) |
| run_rest | comp → stage 2 | ~1 h | **`MLC'26/output_v2/matching_results.tsv`** (submission B) |
| run_ce (starts after stage 2 has finished; one heavy job at a time) | cross-encoder fine-tune + scoring | ~2–2.5 h | |
| run_ce | stage 3 (stage 2 + CE + friend's embedding cosine if ready) | ~45 min | **`MLC'26/output_s3/matching_results.tsv`** (submission C) |

Before submitting any file: the validator must print **PASS** (see the logs). Write CV (OOF F0.5 from the status cell) and LB into LOG.md.
**Account note:** this account needs *Editor* access to the shared `MLC'26` folder, plus a shortcut named `MLC'26` in My Drive.
'''),
    code(SETUP),
    md("## Cell 2: restore data + code (shell cp, ~15 min). Safe to re-run: files already present are skipped (cp -n)."),
    code(r'''
sh(f"""
mkdir -p /content/work /content/out /content/ber/src
cp -n "{D}"/work/train_records.parquet "{D}"/work/test_records.parquet "{D}"/work/train_gt.parquet /content/work/
cp -rn "{D}"/work_v2/* /content/work/
cp "{D}"/code_v2/*.py /content/ber/src/ && cp "{D}"/code_v2/run_rest.sh /content/run_rest.sh
rm -f /content/work/_tmp_*
ls /content/work | wc -l; du -sh /content/work; ls /content/work | grep feat2 | tr '\\n' ' '
""")
'''),
    md("## Cell 3: write the new / updated code (verified locally end to end on a real-data sample) and back it up to Drive"),
    *[writefile("/content/ber/src", f) for f in CODE_FILES],
    writefile("/content", "run_rest.sh"),
    code(r'''
run_ce = r"""#!/bin/bash
# cross-encoder + stage 3. Starts when stage-1 outputs exist; stage 3 waits for stage 2 and for run_rest to finish.
set -e
D="/content/drive/MyDrive/MLC'26"
cd /content/ber/src
export BER_WORK=/content/work BER_OUT=/content/out3 BER_DATA="$D/student_resource/dataset"
SYNC="$D/work_v2"
echo "== waiting for stage 2 to finish (running both at once ran out of RAM) $(date)"
until [ -e /content/work/val_stage2.json ]; do sleep 60; done
while pgrep -f run_rest.sh > /dev/null; do sleep 60; done
( while true; do cp -n /content/work/ce_*.parquet "$SYNC/" 2>/dev/null; sleep 300; done ) &   # finished CE buckets -> Drive every 5 min
echo "== cross-encoder $(date)"
python s8_ce.py count
python s8_ce.py train
cp -r /content/work/ce_model_f0 /content/work/ce_model_f1 "$SYNC/"
python s8_ce.py predict
cp /content/work/ce_*.parquet "$SYNC/"
if [ $(ls "$D"/work_emb/embcos_*.parquet 2>/dev/null | wc -l) = 16 ]; then cp -n "$D"/work_emb/embcos_*.parquet /content/work/; echo "using embedding cosine from work_emb"; else echo "no embedding cosine yet (stage 3 = stage 2 + CE)"; fi
echo "== stage3 $(date)"; python s4_train.py 3
cp /content/work/xgb_s3_* /content/work/val_stage3.json "$SYNC/" || true
python s6_write.py 3
python "$D/student_resource/utils/validate_submission.py" --matching /content/out3/matching_results.tsv \
  --candidate /content/out3/candidate_pairs.tsv --test-dir "$D/student_resource/dataset/test"
mkdir -p "$D/output_s3"; cp /content/out3/*.tsv "$D/output_s3/"
echo "== CE + STAGE 3 DONE $(date)"
"""
open("/content/run_ce.sh", "w").write(run_ce)
sh(f'bash -n /content/run_ce.sh && bash -n /content/run_rest.sh && echo syntax-ok; mkdir -p "{D}/code_v3"; '
   f'cp /content/ber/src/*.py /content/run_rest.sh /content/run_ce.sh "{D}/code_v3/" && ls "{D}/code_v3"')
'''),
    md("## Cell 4: launch both background jobs\n`run_rest` = (skips finished work) -> stage 2 -> submission B. `run_ce` waits until stage 2 is done, then cross-encoder -> stage 3 -> submission C. **Run cells 1-4 only, NOT 'Run all'** (later cells stop or duplicate jobs)."),
    code(r'''
subprocess.Popen("setsid nohup bash /content/run_rest.sh > /content/log_rest.txt 2>&1 < /dev/null &", shell=True)  # setsid: survives kernel restarts
subprocess.Popen("setsid nohup bash /content/run_ce.sh > /content/log_ce.txt 2>&1 < /dev/null &", shell=True)  # setsid: survives kernel restarts
import time; time.sleep(5); sh("pgrep -af 'run_rest|run_ce' | grep -v pgrep")
'''),
    md("## Cell 4b (next account only): run_rest already finished → launch ONLY the cross-encoder + stage 3"),
    code(r'''
subprocess.Popen("setsid nohup bash /content/run_ce.sh > /content/log_ce.txt 2>&1 < /dev/null &", shell=True)  # setsid: survives kernel restarts
import time; time.sleep(5); sh("pgrep -af 'run_ce' | grep -v pgrep; tail -2 /content/log_ce.txt")
'''),
    md("## Cell 5: status (re-run whenever you like)"),
    code(STATUS_MAIN),
    md('''
## Cell 6: per-entity expected-F0.5 decoding (CPU, ~15–30 min, no retraining)
Run it after any stage finishes (set `STAGE` to 1, 2 or 3). It tunes on train OOF and prints a **VERDICT** line.
Upload `MLC'26/output_s{STAGE}_ef/matching_results.tsv` only if the verdict says SUBMIT (and the validator line says PASS).
Don't run it while stage-2/3 training is running (RAM).
'''),
    code(r'''
STAGE = 2
# written as a script file: the folder name MLC'26 contains an apostrophe, which breaks bash -c '...' one-liners
script = f"""#!/bin/bash
set -e
cd /content/ber/src
export BER_WORK=/content/work BER_OUT=/content/out_s{STAGE}
python s9_decode.py {STAGE}
python "{D}/student_resource/utils/validate_submission.py" --matching /content/out_s{STAGE}_ef/matching_results.tsv \
  --candidate /content/out_s{STAGE}_ef/candidate_pairs.tsv --test-dir "{D}/student_resource/dataset/test"
mkdir -p "{D}/output_s{STAGE}_ef"
cp /content/out_s{STAGE}_ef/*.tsv "{D}/output_s{STAGE}_ef/"
cp /content/work/val_decode{STAGE}.json "{D}/work_v2/"
echo DECODE-ALL-DONE
"""
open("/content/run_decode.sh", "w").write(script)
sh("bash -n /content/run_decode.sh && echo syntax-ok")
subprocess.Popen("setsid nohup bash /content/run_decode.sh > /content/log_decode.txt 2>&1 < /dev/null &", shell=True)
# check with:  sh("tail -8 /content/log_decode.txt")
'''),
    md('''
## If something goes wrong
- **A job died with `Killed` in its log** (out of memory): run the clean-stop cell, then re-launch cell 4. Finished work is kept.
  If `run_ce` and stage 2 keep dying together, launch only `run_rest` first, and `run_ce` after `== ALL DONE`.
- **CE is too slow** (eta in `log_ce.txt` > 60 min per fold): stop, then set `os.environ["BER_CE_TRAIN"]="800000"` and relaunch.
  Or narrow the band with `BER_CE_LO=0.05 BER_CE_HI=0.95` (both read by s8_ce.py).
- **Session about to end**: run the save cell below. Next time: cells 1, 2, 3, 4 (everything resumes).
- Send me the last 30 lines of `/content/log_rest.txt` / `/content/log_ce.txt` for any error.
'''),
    code(r'''
# ---- clean stop (kill by PID; never pkill python) ----
sh("for p in $(pgrep -f 'run_[r]est.sh|run_[c]e.sh'); do kill $p; done; sleep 1; "
   "for p in $(pgrep -f 'python s[0-9]_'); do kill $p; done; sleep 3; rm -f /content/work/_tmp_*; pgrep -af python | grep -v colab")
'''),
    code(r'''
# ---- save state to Drive (only new/changed files) ----
sh(f"""cd /content/work
for f in *_norm3.parquet *_idf_*.parquet *_feat2_*.parquet xgb_* p1_* p2_* p3_* comp_* ce_*.parquet val_stage*.json; do
  [ -e "$f" ] || continue
  if [ ! -e "{D}/work_v2/$f" ] || [ $(stat -c%s "$f") != $(stat -c%s "{D}/work_v2/$f") ]; then cp "$f" "{D}/work_v2/"; echo "copied $f"; fi
done
for d in ce_model_f0 ce_model_f1; do [ -e "$d/config.json" ] && cp -rn "$d" "{D}/work_v2/"; done
cp /content/ber/src/*.py /content/run_rest.sh /content/run_ce.sh "{D}/code_v3/"; echo saved""")
'''),
]

STATUS_EMB = r'''
sh("date; tail -15 /content/log_emb.txt; echo; ls /content/work | grep -E 'emb|knn_' | tr '\n' ' '; echo; "
   "pgrep -af 'run_emb|e1_emb' | grep -v pgrep; free -m | head -2; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv")
sh(f'cat "{D}/work_emb/recall_report.txt" 2>/dev/null')
'''

STATUS_EMB = r'''
sh(f"""echo "=== $(date -u +%H:%M) UTC"
if pgrep -f 'run_[e]mb' >/dev/null; then echo "JOB: RUNNING"; else echo "JOB: NOT RUNNING"; fi
grep -E '^> |: [0-9]+s$|==' /content/log_emb.txt | tail -4; tail -1 /content/log_emb.txt
echo "--- errors?"; grep -iE 'error|killed|traceback|out of memory' /content/log_emb.txt | tail -3 || true
echo "--- on Drive (work_emb): chunks $(ls "{D}/work_emb" | grep -c 'test_emb_c[0-7].f16$')/8, test_emb.f16 $(stat -c%s "{D}/work_emb/test_emb.f16" 2>/dev/null || echo missing) bytes, knn_test $(ls "{D}/work_emb" | grep -c knn_test)/8"
free -m | awk 'NR==2{{print "  RAM used " $3 " MB, available " $7 " MB"}}'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | awk '{{print "  GPU " $0}}'""")
'''

EMB = [
    md('''
# PARALLEL job: test-side embeddings for blocking v3
The train side is already done and saved (`MLC'26/work_emb`: train embeddings + kNN; measured recall 0.9503 -> **0.9784**).
This notebook does the same for **test**: encode 11.7M test records with `multilingual-e5-small`, then within-country GPU kNN.
- Output on Drive `MLC'26/work_emb/`: `test_emb.f16` (+ `test_emb_ids.npy`) and `knn_test_0..7.parquet`.
- About the time on a T4: encode **~2.2 h** (the tokenizer on 2 CPUs is the bottleneck), kNN ~20 min. **~2.5 h total.**
- **Resumable:** the encode runs in 8 chunks (~25 min each); every finished chunk is copied to Drive at once. After a crash / dead session /
  another account: run cells 1-4 again and it continues from the last finished chunk. At most one chunk is lost.
- **Run cells 1-4 one by one. Not "Run all".** Then re-run cell 5 for status. Keep the tab open and the laptop awake.

**Account note:** this account needs *Editor* access to the shared `MLC'26` folder, plus a shortcut named `MLC'26` in My Drive.
'''),
    code(SETUP),
    md("## Cell 2: restore what this job needs (test records, code) - about 2 min"),
    code(r'''
sh(f"""
mkdir -p /content/work /content/ber/src "{D}/work_emb"
cp -n "{D}"/work/test_records.parquet /content/work/
cp -n "{D}"/work_emb/test_emb.f16 "{D}"/work_emb/test_emb_ids.npy /content/work/ 2>/dev/null || true
cp -n "{D}"/work_emb/test_emb_c*.f16 /content/work/ 2>/dev/null || true   # finished chunks from an earlier session
cp -n "{D}"/work_emb/knn_test_*.parquet /content/work/ 2>/dev/null || true
ls -la /content/work; df -h /content | tail -1
""")
'''),
    md("## Cell 3: write the code (verified locally end to end on a real-data sample)"),
    writefile("/content/ber/src", "common.py"),
    writefile("/content/ber/src", "s2_block.py"),
    writefile("/content/ber/src", "e1_emb.py"),
    code(r'''
run_emb = r"""#!/bin/bash
set -e
D="/content/drive/MyDrive/MLC'26"
cd /content/ber/src
OUT="$D/work_emb"
export BER_WORK=/content/work BER_EMB_MODEL="$D/models/multilingual-e5-small" BER_EMB_SYNC="$OUT"
echo "== encode test $(date)"
python e1_emb.py encode test
[ "$(stat -c%s "$OUT/test_emb.f16" 2>/dev/null)" = "$(stat -c%s /content/work/test_emb.f16)" ] || cp /content/work/test_emb.f16 /content/work/test_emb_ids.npy "$OUT/"
[ "$(stat -c%s "$OUT/test_emb.f16")" = "$(stat -c%s /content/work/test_emb.f16)" ] && rm -f "$OUT"/test_emb_c*.f16   # full file safe on Drive -> drop the chunks
echo "== knn test $(date)"
python e1_emb.py knn test
cp /content/work/knn_test_*.parquet "$OUT/"
echo "== EMB TEST DONE $(date)"
"""
open("/content/run_emb.sh", "w").write(run_emb)
sh('bash -n /content/run_emb.sh && echo syntax-ok')
'''),
    md("## Cell 4: launch in the background"),
    code(r'''
subprocess.Popen("setsid nohup bash /content/run_emb.sh > /content/log_emb.txt 2>&1 < /dev/null &", shell=True)  # setsid: survives kernel restarts
import time; time.sleep(10); sh("pgrep -af 'run_emb' | grep -v pgrep; tail -2 /content/log_emb.txt")
'''),
    md("## Cell 5: status (re-run any time)"),
    code(STATUS_EMB),
    md('''
## Done when
Cell 5 shows `== EMB TEST DONE`, the Drive `test_emb.f16` at **~9.0 GB** (matching the local file), and `knn_test 8/8`. Then delete the runtime.

## If something goes wrong
- `JOB: NOT RUNNING` without `EMB TEST DONE`: re-run cell 4 (finished steps are skipped). Send the last 30 lines of `/content/log_emb.txt` if it fails again.
- GPU out of memory in kNN: in cell 3 change `QB = 1024` to `QB = 512` in e1_emb.py, re-run cell 3, then cell 4.
'''),
    code(r'''
# ---- clean stop ----
sh("for p in $(pgrep -f 'run_[e]mb.sh'); do kill $p; done; sleep 1; for p in $(pgrep -f 'python [e]1_emb'); do kill $p; done; sleep 2; pgrep -af python | grep -v colab")
'''),
]

STATUS_V3 = r'''
sh("date; tail -8 /content/log_v3.txt; echo; ls /content/work | grep -E 'feat3|_v3' | tr '\n' ' '; echo; "
   "pgrep -af 'run_v3|python s' | grep -v pgrep; free -m | head -2; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
import json
for s in (1, 4, 5, 2, 3):
    p = f"/content/work/val_stage{s}_v3.json"
    if os.path.exists(p):
        d = json.load(open(p)); print(f"v3 stage {s}: OOF F0.5 {d['f05']:.4f} at t={d['t']}  by country", {k: round(v, 4) for k, v in d['by_country'].items()})
'''

V3 = [
    md('''
# V3 pipeline: keys + embedding-kNN candidates (recall 0.950 -> 0.978), then stages 1-2-3
Needs on Drive: `work_v2` (v2 state), `work_emb` with **train and test** embeddings + kNN (test side from `PARALLEL_embeddings.ipynb`),
and ideally the cross-encoder fold models `work_v2/ce_model_f0|f1` from the MAIN run (otherwise they are trained here, ~1 h).
Everything v3 writes has a `_v3` tag / `feat3` name, so **no v2 file is ever overwritten**.

| step | about the time (T4) | output to submit |
|---|---|---|
| restore (~35 GB) | 30 min | |
| embedding cosine for all pairs (train + test) | 40 min | |
| v3 features: new pairs only + merge (16 buckets) | 1.5 h | |
| stage 1 | 1.3 h | **`MLC'26/output_v3s1/`** (submit first) |
| cross-encoder (reuses models if on Drive) + stage 4 = stage 1 + CE, no stacking | 2 h | **`MLC'26/output_v3s4/`** |
| extra: fold-consistent comp + stage 2, stage 3 | 2 h | `output_v3s2/`, `output_v3s3/` (stacked; v2 stacking lost 0.004 on LB, so treat as experiments) |

It won't all fit in one ~5 h session. Small essentials are saved to Drive (`work_v3/`: models, scores, stage-1 predictions, CE scores, the helper's test features);
the big train features are NOT saved (Drive quota is per creating account, ~15 GB) and are rebuilt (~70 min) if a second account has to take over. On the next account run cells 1-4 again and it resumes.
**Run cells 1-4 one by one. Not "Run all".** Keep the tab open, laptop awake. Upload a file only after the validator printed **PASS** in `log_v3.txt`.
'''),
    code(SETUP),
    md("## Cell 2: restore (shell cp, ~30 min; re-runnable, skips files already present)"),
    code(r'''
sh(f"""
mkdir -p /content/work /content/out /content/ber/src "{D}/work_v3"
cp -n "{D}"/work/train_records.parquet "{D}"/work/test_records.parquet "{D}"/work/train_gt.parquet /content/work/
cp -n "{D}"/work_v2/*_norm2.parquet "{D}"/work_v2/*_norm3.parquet "{D}"/work_v2/*_idf_*.parquet /content/work/
cp -n "{D}"/work_v2/train_cand2_[0-7].parquet "{D}"/work_v2/test_cand2_[0-7].parquet "{D}"/work_v2/*_feat2_*.parquet /content/work/
cp -rn "{D}"/work_v2/ce_model_f0 "{D}"/work_v2/ce_model_f1 /content/work/ 2>/dev/null || echo "no CE models on Drive yet (they will be trained)"
for f in train_emb.f16 train_emb_ids.npy test_emb.f16 test_emb_ids.npy; do cp -n "{D}/work_emb/$f" /content/work/ || echo "MISSING $f"; done
cp -n "{D}"/work_emb/knn_*.parquet /content/work/
cp -rn "{D}"/work_v3/* /content/work/ 2>/dev/null || true
rm -rf /content/work/_tmp* /content/work/*.part   # leftovers of an interrupted copy are never used
echo "knn files: $(ls /content/work | grep -c knn_) (need 16)   emb files: $(ls /content/work | grep -c _emb) (need 4)"
du -sh /content/work; df -h /content | tail -1
""")
'''),
    md("## Cell 3: write the code (verified locally end to end on a real-data sample)"),
    *[writefile("/content/ber/src", f) for f in CODE_FILES],
    code(r'''
run_v3 = r"""#!/bin/bash
set -e
D="/content/drive/MyDrive/MLC'26"
cd /content/ber/src
export BER_WORK=/content/work BER_DATA="$D/student_resource/dataset"
SYNC="$D/work_v3"; mkdir -p "$SYNC"
VAL="$D/student_resource/utils/validate_submission.py"; TEST="$D/student_resource/dataset/test"
save() { # copy to Drive under a temp name, rename only when complete: a session dying mid-copy never leaves a truncated file
  for f in "$@"; do [ -e "$f" ] || continue; b=$(basename "$f")
    [ "$(du -sb "$f" | cut -f1)" = "$(du -sb "$SYNC/$b" 2>/dev/null | cut -f1)" ] && continue
    rm -rf "$SYNC/$b.part"
    if cp -r "$f" "$SYNC/$b.part"; then rm -rf "$SYNC/$b"; mv "$SYNC/$b.part" "$SYNC/$b"; else echo "WARN: Drive copy failed for $b"; fi
  done; }
submit() { # stage outdir name
  BER_OUT=$2 python s6_write.py $1
  python "$VAL" --matching $2/matching_results.tsv --test-dir "$TEST"   # candidate file (~108M pairs) OOM-kills the validator; matches come from candidates by construction
  mkdir -p "$D/$3"; cp $2/*.tsv "$D/$3/"; echo "== SUBMISSION READY: MLC'26/$3 $(date)"; }
[ $(ls /content/work | grep -c knn_) = 16 ] || { echo "ERROR: need 16 knn files (train + test) in /content/work"; exit 1; }
nfeat() { ls /content/work | grep -c "^$1_feat3_[0-7].parquet$"; }
tidy() { # free local disk: delete inputs whose next step is finished (Colab disk ~112 GB fills up otherwise)
  cd /content/work
  [ $(ls | grep -c '^embcos3_train_[0-7].parquet$') = 8 ] && rm -f train_emb.f16
  [ $(ls | grep -c '^embcos3_test_[0-7].parquet$') = 8 ] && rm -f test_emb.f16
  for j in 0 1 2 3 4 5 6 7; do [ -e train_feat3_$j.parquet ] && rm -f train_feat2_$j.parquet; [ -e test_feat3_$j.parquet ] && rm -f test_feat2_$j.parquet; done
  [ $(nfeat test) = 8 ] && rm -f test_emb.f16
  cd /content/ber/src; df -h /content | tail -1; }
if [ "${BER_V3_MODE:-full}" = "test" ]; then   # helper account: build only the test half, then stop
  echo "== TEST-HALF MODE: cosine + feat3 for test $(date)"
  python s10_v3.py cos test; tidy
  for j in 0 1 2 3 4 5 6 7; do cp -n "$SYNC/test_feat3_$j.parquet" /content/work/ 2>/dev/null || true
    python s10_v3.py build test $j; save /content/work/test_feat3_$j.parquet; tidy; done
  echo "== TEST FEAT3 DONE $(date)"; exit 0
fi
echo "== train: cosine + feat3 $(date)"
[ $(nfeat train) = 8 ] || python s10_v3.py cos train
tidy
for j in 0 1 2 3 4 5 6 7; do python s10_v3.py build train $j; tidy; done   # not saved to Drive (8 GB; rebuilt in ~70 min if needed)
echo "== test: take buckets the helper account already saved, build the rest $(date)"
cp -n "$SYNC"/test_feat3_*.parquet /content/work/ 2>/dev/null || true
[ $(nfeat test) = 8 ] || python s10_v3.py cos test
tidy
for j in 0 1 2 3 4 5 6 7; do cp -n "$SYNC/test_feat3_$j.parquet" /content/work/ 2>/dev/null || true
  python s10_v3.py build test $j; tidy; done   # only the helper account saves test feat3 (quota)
python s10_v3.py report
export BER_FEAT=feat3 BER_TAG=_v3
echo "== stage1 $(date)"; python s4_train.py 1
save /content/work/xgb_s1_v3_* /content/work/val_stage1_v3.json /content/work/p1_v3_*.parquet
[ -e "$D/output_v3s1/matching_results.tsv" ] || submit 1 /content/out_v3s1 output_v3s1
echo "== cross-encoder $(date)"; python s8_ce.py all
save /content/work/ce_model_f0 /content/work/ce_model_f1 /content/work/ce_v3_*.parquet
echo "== stage4 (stage 1 + cross-encoder, no stacking) $(date)"; python s4_train.py 4
save /content/work/xgb_s4_v3_* /content/work/val_stage4_v3.json
[ -e "$D/output_v3s4/matching_results.tsv" ] || submit 4 /content/out_v3s4 output_v3s4
echo "== address add-on features (house-number noise) $(date)"
for sp in train test; do python s14_addr.py $sp; done
echo "== stage5 (stage 4 + address features) $(date)"; python s4_train.py 5
save /content/work/xgb_s5_v3_* /content/work/val_stage5_v3.json
[ -e "$D/output_v3s5/matching_results.tsv" ] || submit 5 /content/out_v3s5 output_v3s5
echo "== comp (fold-consistent) $(date)"; python s5_comp.py
echo "== stage2 $(date)"; python s4_train.py 2
save /content/work/xgb_s2_v3_* /content/work/val_stage2_v3.json
[ -e "$D/output_v3s2/matching_results.tsv" ] || submit 2 /content/out_v3s2 output_v3s2
echo "== stage3 $(date)"; python s4_train.py 3
save /content/work/xgb_s3_v3_* /content/work/val_stage3_v3.json
[ -e "$D/output_v3s3/matching_results.tsv" ] || submit 3 /content/out_v3s3 output_v3s3
echo "== V3 ALL DONE $(date)"
"""
open("/content/run_v3.sh", "w").write(run_v3)
sh(f'bash -n /content/run_v3.sh && echo syntax-ok; mkdir -p "{D}/code_v3"; cp /content/ber/src/*.py /content/run_v3.sh "{D}/code_v3/"')
'''),
    md("## Cell 3b (optional, BEFORE cell 4): expansion diagnostic, ~10 min CPU\n"
       "Of the true pairs the v3 candidates still miss (2.2%), how many look like a sibling duplicate we already predict? "
       "Needs the files cell 2 restored (train embeddings, cand2, kNN, p1_v3_train). Send the printed lines to the team."),
    code(r'''
sh("cd /content/ber/src && BER_WORK=/content/work BER_TAG=_v3 python s12_expand_diag.py 2>&1 | tail -8")
'''),
    md("## Cell 4: launch (background, survives kernel restarts)"),
    code(r'''
subprocess.Popen("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup bash /content/run_v3.sh > /content/log_v3.txt 2>&1 < /dev/null &", shell=True)
import time; time.sleep(20); sh("pgrep -af 'run_v3' | grep -v pgrep; tail -3 /content/log_v3.txt")
'''),
    md("## Cell 4c (HELPER account only, instead of cell 4): build only the TEST half of the v3 features, in parallel with the main account\n"
       "Run cells 1, 2, 3, then this. It stops by itself after `== TEST FEAT3 DONE` (~1.5 h). The main account picks the saved buckets up from Drive."),
    code(r'''
subprocess.Popen("BER_V3_MODE=test PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup bash /content/run_v3.sh > /content/log_v3.txt 2>&1 < /dev/null &", shell=True)
import time; time.sleep(20); sh("pgrep -af 'run_v3' | grep -v pgrep; tail -3 /content/log_v3.txt")
'''),
    md("## Cell 5: status (re-run any time)"),
    code(STATUS_V3),
    md('''
## If something goes wrong
- `Killed` / job gone: re-run cell 4. Every finished step is skipped (feat3 buckets, models, predictions).
- Session ending: everything finished is already in `MLC'26/work_v3` (copied as it goes). Next account: cells 1-4.
- `ERROR: need 16 knn files`: the test side (`PARALLEL_embeddings.ipynb`) hasn't finished yet.
- Send the last 30 lines of `/content/log_v3.txt` for any error.
'''),
    code(r'''
# ---- clean stop ----
sh("for p in $(pgrep -f 'run_[v]3.sh'); do kill $p; done; sleep 1; for p in $(pgrep -f 'python s[0-9]'); do kill $p; done; sleep 3; rm -f /content/work/_tmp*; pgrep -af python | grep -v colab")
'''),
]

for name, cells in (("MAIN_pipeline.ipynb", MAIN), ("PARALLEL_embeddings.ipynb", EMB), ("V3_pipeline.ipynb", V3)):
    (Path(__file__).parent / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name, len(cells), "cells")
