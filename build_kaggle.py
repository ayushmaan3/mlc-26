"""Builds KAGGLE_run.ipynb (full-data stage 5 + second model + blend) from the verified code in code_v2/."""
import json
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "code_v2"
CODE_FILES = ["common.py", "s0_load.py", "s1_normalize.py", "s2_block.py", "s3_features.py", "s4_train.py", "s5_comp.py",
              "s6_write.py", "s8_ce.py", "s9_decode.py", "s10_v3.py", "s11_tune.py", "s13_errors.py", "s14_addr.py",
              "s15_blend.py", "s16_idcomp.py", "e1_emb.py", "s17_ft_retrieval.py", "s18_ftcos.py", "s19_preflight.py", "s20_rules.py", "s21_efdecode.py", "s22_stack.py", "s23_ceband.py"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def writefile(path, name):
    return code(f"%%writefile {path}/{name}\n" + (C / name).read_text(encoding="utf-8"))


RUN = r'''#!/bin/bash
# Full-data stage 5 (+ a second, differently-configured stage-5 model and their blend) on Kaggle (31 GB RAM, 4 CPUs, T4).
set -e
W=/tmp/work; OUT=/kaggle/working; mkdir -p $W $OUT/state
CORE=$(dirname "$(find /kaggle/input -name train_records.parquet | head -1)")
EMB=$(dirname "$(find /kaggle/input -name train_emb.f16 | head -1)")
V3=$(dirname "$(find /kaggle/input -name ce_v3_train_0.parquet | head -1)")
echo "inputs: core=$CORE emb=$EMB v3=$V3"
for d in "$CORE" "$EMB" "$V3"; do for f in "$d"/*; do [ -f "$f" ] && ln -sf "$f" $W/; done; done
for m in ce_model_f0 ce_model_f1; do d=$(find /kaggle/input -type d -name $m | head -1); [ -n "$d" ] && ln -sfn "$d" $W/$m || true; done   # not needed by stage 5/6
echo "linked $(ls $W | wc -l) inputs"
for f in train_records.parquet test_records.parquet train_gt.parquet train_norm2.parquet test_norm3.parquet train_feat2_0.parquet \
         train_emb.f16 test_emb.f16 knn_test_7.parquet ce_v3_test_7.parquet addr4_test_7.parquet test_source1.tsv validate_submission.py; do
  [ -e $W/$f ] || { echo "MISSING $f"; exit 1; }; done
cd $OUT/src
export BER_WORK=$W BER_OUT=$OUT BER_FEAT=feat3 BER_TAG=_v3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BER_NPIECE=8
free -g | head -2
VAL="python $W/validate_submission.py --test-dir $CORE --matching"
echo "== cosine $(date)"; python s10_v3.py cos train; python s10_v3.py cos test
echo "== feat3 $(date)"
for s in train test; do for j in 0 1 2 3 4 5 6 7; do python s10_v3.py build $s $j; done; done
python s10_v3.py report
echo "== stage5 FULL DATA $(date)"
python s4_train.py 5 || { echo "== stage5 full data failed (likely RAM): retrying the unfinished fold(s) on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 5; }
cp $W/val_stage5_v3.json $W/xgb_s5_v3_* $OUT/state/
python s15_blend.py v3s5full 5
$VAL $OUT/v3s5full/matching_results.tsv | tail -1; $VAL $OUT/v3s5full_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: v3s5full + v3s5full_fr90 $(date)"
echo "== stage6: second stage-5 model (511 leaves, other seed/columns) $(date)"
export BER_LEAVES=511 BER_SEED=1 BER_COLS=0.7 BER_MCW=40
python s4_train.py 6 || { echo "== stage6 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 6; }
cp $W/val_stage6_v3.json $W/xgb_s6_v3_* $OUT/state/
python s15_blend.py blend56 5 6
$VAL $OUT/blend56/matching_results.tsv | tail -1; $VAL $OUT/blend56_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend56 + blend56_fr90 $(date)"
mkdir -p $OUT/state/preds; cp $W/p5_v3_test_*.parquet $W/p6_v3_test_*.parquet $W/val_blend_*.json $OUT/state/preds/ 2>/dev/null || true
du -sh $OUT; echo "== KAGGLE RUN DONE $(date)"
'''

RUN_B = RUN.split('echo "== stage5 FULL DATA')[0] + r'''echo "== per-record competition features $(date)"
python s16_idcomp.py train; python s16_idcomp.py test
echo "== stage7 = stage 5 + competition features, FULL DATA $(date)"
python s4_train.py 7 || { echo "== stage7 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 7; }
cp $W/val_stage7_v3.json $W/xgb_s7_v3_* $OUT/state/
python s15_blend.py v3s7 7
$VAL $OUT/v3s7/matching_results.tsv | tail -1; $VAL $OUT/v3s7_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: v3s7 + v3s7_fr90 $(date)"
echo "== stage8: second model with competition features (511 leaves, other seed/columns) $(date)"
export BER_LEAVES=511 BER_SEED=1 BER_COLS=0.7 BER_MCW=40
python s4_train.py 8 || { echo "== stage8 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 8; }
python s15_blend.py blend78 7 8
$VAL $OUT/blend78/matching_results.tsv | tail -1; $VAL $OUT/blend78_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend78 + blend78_fr90 $(date)"
mkdir -p $OUT/state/preds; cp $W/p7_v3_test_*.parquet $W/p8_v3_test_*.parquet $W/val_blend_*.json $OUT/state/preds/ 2>/dev/null || true
du -sh $OUT; echo "== KAGGLE RUN B DONE $(date)"
'''


FT_BLOCK = r'''FT=$(dirname "$(find /kaggle/input -name model.safetensors -path '*ftmodel*' | head -1)")
echo "fine-tuned e5: $FT"
for f in $(find /kaggle/input -name 'knnft_*.parquet'); do ln -sf "$f" $W/; done
[ -e "$FT/model.safetensors" ] || [ -e $W/knnft_test_7.parquet ] || { echo "MISSING fine-tuned model: add the mlc26-ftmodel dataset"; exit 1; }
if [ ! -e $W/knnft_test_7.parquet ]; then
  # the model was saved by transformers 5.x on Colab: prove it loads here (and is not randomly initialised) before 6 h of work
  CHECK="from transformers import AutoModel, AutoTokenizer; import torch, transformers
t = AutoTokenizer.from_pretrained('/tmp/e5ft'); m = AutoModel.from_pretrained('/tmp/e5ft').eval()
x = t(['query: sunrise pizza | 12 main street springfield', 'query: sunrise pizza llc | 12 main st springfield',
       'query: delhi dental clinic | 44 mg road pune'], padding=True, return_tensors='pt')
with torch.no_grad(): h = m(**x).last_hidden_state
e = torch.nn.functional.normalize((h * x['attention_mask'][..., None]).sum(1) / x['attention_mask'].sum(1, keepdim=True), dim=1)
same, diff = float(e[0] @ e[1]), float(e[0] @ e[2])
print(f'fine-tuned e5 loads (transformers {transformers.__version__}): cos same-business {same:.3f}, different {diff:.3f}')
assert same > 0.8 and same - diff > 0.05, 'fine-tuned model looks broken'"
  rm -rf /tmp/e5ft; mkdir -p /tmp/e5ft; cp -r $FT/* /tmp/e5ft/
  if ! python -c "$CHECK"; then
    ORIG=$(dirname "$(find /kaggle/input -name model.safetensors -path '*multilingual-e5-small*' | grep -v ftmodel | head -1)")
    echo "fine-tuned dir did not load: using config + tokenizer of the original e5 ($ORIG) with the fine-tuned weights"
    rm -rf /tmp/e5ft; mkdir -p /tmp/e5ft; cp -r $ORIG/* /tmp/e5ft/; cp $FT/model.safetensors /tmp/e5ft/model.safetensors
    python -c "$CHECK"
  fi
  FT=/tmp/e5ft
  echo "== encode train + test with the fine-tuned e5, one per GPU $(date)"
  F=/tmp/ft; mkdir -p $F; ln -sf $W/train_records.parquet $W/test_records.parquet $F/
  BER_WORK=$F BER_EMB_MODEL=$FT CUDA_VISIBLE_DEVICES=0 python e1_emb.py encode train > $OUT/ft_enc_train.log 2>&1 & P1=$!
  BER_WORK=$F BER_EMB_MODEL=$FT CUDA_VISIBLE_DEVICES=1 python e1_emb.py encode test > $OUT/ft_enc_test.log 2>&1 & P2=$!
  wait $P1; wait $P2; tail -n 2 $OUT/ft_enc_train.log $OUT/ft_enc_test.log; rm -f $F/*_emb_c*.f16
  echo "== kNN@20 with the fine-tuned e5 $(date)"
  BER_WORK=$F CUDA_VISIBLE_DEVICES=0 python e1_emb.py knn train > $OUT/ft_knn_train.log 2>&1 & P1=$!
  BER_WORK=$F CUDA_VISIBLE_DEVICES=1 python e1_emb.py knn test > $OUT/ft_knn_test.log 2>&1 & P2=$!
  wait $P1; wait $P2
  mkdir -p $OUT/state/knnft
  for s in train test; do for j in 0 1 2 3 4 5 6 7; do
    cp $F/knn_${s}_${j}.parquet $OUT/state/knnft/knnft_${s}_${j}.parquet; ln -sf $OUT/state/knnft/knnft_${s}_${j}.parquet $W/; done; done
  rm -rf $F
fi
'''

RUN_C = (RUN.replace('do [ -f "$f" ] && ln', 'do case "$f" in *addr4_*) continue;; esac; [ -f "$f" ] && ln')   # rebuilt below
         .replace(" addr4_test_7.parquet", "")
         .replace('echo "== cosine', FT_BLOCK + 'echo "== cosine')
         .replace("python s10_v3.py report\n", 'python s10_v3.py report\necho "== address add-on for the new union $(date)"\n'
                  "python s14_addr.py train; python s14_addr.py test\n")
         .replace("v3s5full", "v3s5ft").replace("blend56", "blend56ft")
         .replace("# Full-data stage 5", "# Run C: + candidates from the fine-tuned e5 (s17), then full-data stage 5")
         .replace("KAGGLE RUN DONE", "KAGGLE RUN C DONE"))
assert RUN_C.count("-path '*ftmodel*'") == 1 and "case \"$f\"" in RUN_C and "s14_addr.py train" in RUN_C


CELLS = [
    md("""
# Kaggle run: full-data stage 5 + second model + blend (runs in the background)
**Setup (once, right-hand panel):**
1. **Add Input -> Your Work -> Datasets**: add `mlc26-core`, `mlc26-emb`, `mlc26-v3`.
2. **Session options: Accelerator GPU T4 x2, Internet On.**
3. Click **Save Version -> Save & Run All (Commit)**. It runs in the background (browser and laptop can be off), up to 12 h.

**Outputs** (Output tab of the saved version; the log is `log_kaggle.txt`):
- `v3s5full_fr90/matching_results.tsv`: submit first (full-data stage 5, t 0.80 / France 0.90), ready after ~3 h
- `blend56_fr90/matching_results.tsv`: stage 5 + a second model averaged (usually a little better), ~5 h
- `v3s5full/`, `blend56/`: the same at the OOF-best threshold; `candidates/candidate_pairs.tsv` for the final zip.
The OOF score of each is in the log: `OOF best t=... F0.5=...`.
"""),
    code(r'''
import subprocess, os
print(subprocess.run("pip -q install rapidfuzz anyascii 2>&1 | tail -1; nproc; free -g | head -2; nvidia-smi -L; find /kaggle/input -maxdepth 3 -type d | head -20",
                     shell=True, capture_output=True, text=True).stdout)
os.makedirs("/kaggle/working/src", exist_ok=True)
'''),
    *[writefile("/kaggle/working/src", f) for f in CODE_FILES],
    code("run = " + repr(RUN) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
         'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'),
    code(r'''
# runs the whole chain; output is streamed to the notebook and to /kaggle/working/log_kaggle.txt
p = subprocess.Popen("bash /kaggle/working/run_kaggle.sh 2>&1 | tee /kaggle/working/log_kaggle.txt", shell=True)
p.wait()
print("exit code", p.returncode)
'''),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
(HERE / "KAGGLE_run.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle.sh").write_text(RUN, encoding="utf-8", newline="\n")
print("wrote KAGGLE_run.ipynb", len(CELLS), "cells")

CELLS_B = [md(CELLS[0]["source"].replace("full-data stage 5 + second model + blend", "stage 7 = stage 5 + per-record competition features")
             .replace("v3s5full_fr90", "v3s7_fr90").replace("blend56_fr90", "blend78_fr90").replace("v3s5full/", "v3s7/").replace("blend56/", "blend78/"))]
CELLS_B += CELLS[1:-2]
CELLS_B.append(code("run = " + repr(RUN_B) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                    'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_B.append(CELLS[-1])
nbB = dict(nb, cells=CELLS_B)
(HERE / "KAGGLE_runB.ipynb").write_text(json.dumps(nbB, indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_B.sh").write_text(RUN_B, encoding="utf-8", newline="\n")
print("wrote KAGGLE_runB.ipynb", len(CELLS_B), "cells")

CELLS_C = [md(CELLS[0]["source"].replace("full-data stage 5 + second model + blend", "run C = fine-tuned e5 candidates + full-data stage 5")
             .replace("`mlc26-core`, `mlc26-emb`, `mlc26-v3`", "`mlc26-core`, `mlc26-emb`, `mlc26-v3`, `mlc26-ftmodel`")
             .replace("v3s5full", "v3s5ft").replace("blend56", "blend56ft")
             .replace("ready after ~3 h", "ready after ~5 h").replace(", ~5 h", ", ~6.5 h"))]
CELLS_C += CELLS[1:-2]
CELLS_C.append(code("run = " + repr(RUN_C) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                    'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_C.append(CELLS[-1])
(HERE / "KAGGLE_runC.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_C), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_C.sh").write_text(RUN_C, encoding="utf-8", newline="\n")
print("wrote KAGGLE_runC.ipynb", len(CELLS_C), "cells")


# Run D = run B's stage 7/8 (competition features) on run C's bigger union. Reuses C's kNN files (Output of KAGGLE_runC
# added as input) -> no re-encoding; fails fast instead of silently spending 2.5 h re-encoding.
KNNFT_LINK = r"""for f in $(find /kaggle/input -name 'knnft_*.parquet'); do ln -sf "$f" $W/; done
ls $W/knnft_train_{0..7}.parquet $W/knnft_test_{0..7}.parquet > /dev/null || { echo "MISSING knnft_*: add the Output of KAGGLE_runC as input"; exit 1; }
echo "reusing fine-tuned kNN of run C: $(ls $W/knnft_*.parquet | wc -l) files"
"""
RUN_D = (RUN_C.replace(FT_BLOCK, KNNFT_LINK).split('echo "== stage5 FULL DATA')[0]
         + RUN_B.split('python s10_v3.py report\n')[1]
         ).replace("v3s7", "v3s7ft").replace("blend78", "blend78ft").replace("KAGGLE RUN B DONE", "KAGGLE RUN D DONE") \
          .replace("# Run C: + candidates from the fine-tuned e5 (s17), then full-data stage 5", "# Run D: run C union + stage 7/8")
assert KNNFT_LINK in RUN_D and "s14_addr.py train" in RUN_D and "s16_idcomp.py train" in RUN_D and "e1_emb.py" not in RUN_D
CELLS_D = [md(CELLS[0]["source"].replace("full-data stage 5 + second model + blend", "run D = run C candidates + stage 7 (competition features)")
             .replace("`mlc26-core`, `mlc26-emb`, `mlc26-v3`", "`mlc26-core`, `mlc26-emb`, `mlc26-v3`, and under **Your Work -> Notebooks** the output of `KAGGLE_runC`")
             .replace("v3s5full", "v3s7ft").replace("blend56", "blend78ft")
             .replace("ready after ~3 h", "ready after ~3 h").replace(", ~5 h", ", ~4 h"))]
CELLS_D += CELLS[1:-2]
CELLS_D.append(code("run = " + repr(RUN_D) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                    'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_D.append(CELLS[-1])
(HERE / "KAGGLE_runD.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_D), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_D.sh").write_text(RUN_D, encoding="utf-8", newline="\n")
print("wrote KAGGLE_runD.ipynb", len(CELLS_D), "cells")


# Run E = run D + fold-wise fine-tuned e5 cosine features (s18_ftcos.py, Fable review 2). Same inputs as D.
E_BLOCK = r"""E5=$(dirname "$(find /kaggle/input -name model.safetensors -path '*multilingual-e5-small*' | grep -v -e ftmodel -e e5_ft | head -1)")
if [ ! -e "$E5/model.safetensors" ]; then
  echo "original e5 not in the inputs: downloading intfloat/multilingual-e5-small (MIT) from Hugging Face"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', local_dir='/tmp/e5orig', allow_patterns=['*.json', 'model.safetensors', '*.model'])"
  E5=/tmp/e5orig
fi
echo "original e5: $E5"; export BER_E5=$E5
echo "== fine-tune the two fold models, one per GPU $(date)"
CUDA_VISIBLE_DEVICES=0 python s18_ftcos.py train 0 > $OUT/ftE_m0.log 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=1 python s18_ftcos.py train 1 > $OUT/ftE_m1.log 2>&1 & P2=$!
wait $P1 || { tail -30 $OUT/ftE_m0.log; exit 1; }
wait $P2 || { tail -30 $OUT/ftE_m1.log; exit 1; }
grep -h "recall@\|fine-tune m" $OUT/ftE_m0.log $OUT/ftE_m1.log
echo "== encode every record with both fold models $(date)"
AV=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9); echo "free disk on /tmp: ${AV} GB (both splits at once need ~37 GB)"
if [ "${AV:-0}" -ge 60 ]; then
  python s18_ftcos.py encode train > $OUT/ftE_enc_train.log 2>&1 & P1=$!
  python s18_ftcos.py encode test > $OUT/ftE_enc_test.log 2>&1 & P2=$!
  wait $P1 || { tail -30 $OUT/ftE_enc_train.log; exit 1; }
  wait $P2 || { tail -30 $OUT/ftE_enc_test.log; exit 1; }
  tail -n 1 $OUT/ftE_enc_train.log $OUT/ftE_enc_test.log
  echo "== fine-tuned cosine features $(date)"
  python s18_ftcos.py feats train; python s18_ftcos.py feats test; rm -f $W/*_embft*.f16
else
  for s in train test; do
    python s18_ftcos.py encode $s > $OUT/ftE_enc_$s.log 2>&1 || { tail -30 $OUT/ftE_enc_$s.log; exit 1; }
    tail -n 1 $OUT/ftE_enc_$s.log; python s18_ftcos.py feats $s; rm -f $W/${s}_embft*.f16
  done
fi
mkdir -p $OUT/state/ftE; cp -r $W/e5ft_m0 $W/e5ft_m1 $OUT/state/ftE/
"""
E_TAIL = r"""echo "== stage9 = stage 7 + fine-tuned cosine features, FULL DATA $(date)"
python s4_train.py 9 || { echo "== stage9 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 9; }
cp $W/val_stage9_v3.json $W/xgb_s9_v3_* $OUT/state/
python s15_blend.py v3s9 9
for v in v3s9 v3s9_fr90 v3s9_t80; do $VAL $OUT/$v/matching_results.tsv | tail -1; done
echo "== SUBMISSION READY: v3s9 (+ _fr90, _t80) $(date)"
echo "== stage10: second model (511 leaves, other seed/columns) $(date)"
export BER_LEAVES=511 BER_SEED=1 BER_COLS=0.7 BER_MCW=40
python s4_train.py 10 || { echo "== stage10 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 10; }
python s15_blend.py blend910 9 10
for v in blend910 blend910_fr90 blend910_t80; do $VAL $OUT/$v/matching_results.tsv | tail -1; done
echo "== SUBMISSION READY: blend910 (+ _fr90, _t80) $(date)"
mkdir -p $OUT/state/preds; cp $W/p9_v3_test_*.parquet $W/p10_v3_test_*.parquet $W/val_blend_*.json $OUT/state/preds/ 2>/dev/null || true
du -sh $OUT; echo "== KAGGLE RUN E DONE $(date)"
"""
E_PREFLIGHT = r"""for f in $(find /kaggle/input -name 'knnft_*.parquet'); do ln -sf "$f" $W/; done
E5=$(dirname "$(find /kaggle/input -name model.safetensors -path '*multilingual-e5-small*' | grep -v -e ftmodel -e e5_ft | head -1)")
echo "== PREFLIGHT: inputs, GPUs, e5, disk $(date)"
E5="$E5" python s19_preflight.py
"""
NL = chr(10)
_E_BODY = E_BLOCK.split(NL, 1)[1]   # E_BLOCK without its first line (the E5 lookup, now in the preflight)
assert _E_BODY.startswith('if [ ! -e "$E5/model.safetensors" ]')
RUN_E = (RUN_D.split('echo "== stage7')[0].replace(KNNFT_LINK, E_PREFLIGHT + _E_BODY) + E_TAIL) \
    .replace("# Run D: run C union + stage 7/8", "# Run E: run D + fold-wise fine-tuned e5 cosine features (stage 9/10)")
_i = RUN_E.index("for f in train_records.parquet")   # the old stop-at-first-missing check: the preflight lists everything
_END = "exit 1; }; done" + NL
RUN_E = RUN_E[:_i] + RUN_E[RUN_E.index(_END, _i) + len(_END):]
assert 'echo "MISSING' not in RUN_E and RUN_E.index("s19_preflight") < RUN_E.index("s18_ftcos.py train 0")
assert RUN_E.count("s18_ftcos.py train 0") == 1 and "s16_idcomp.py test" in RUN_E and RUN_E.index("s18_ftcos") < RUN_E.index("== cosine")
CELLS_E = [md(CELLS[0]["source"].replace("full-data stage 5 + second model + blend", "run E = run D + fold-wise fine-tuned e5 cosine features")
             .replace("`mlc26-core`, `mlc26-emb`, `mlc26-v3`", "`mlc26-core`, `mlc26-emb`, `mlc26-v3`, and under **Your Work -> Notebooks** the output of `KAGGLE_runC`")
             .replace("v3s5full_fr90", "v3s9_fr90").replace("blend56_fr90", "blend910_fr90").replace("v3s5full/", "v3s9/ and v3s9_t80/").replace("blend56/", "blend910/ and blend910_t80/")
             .replace("ready after ~3 h", "ready after ~5 h").replace(", ~5 h", ", ~6 h"))]
CELLS_E += CELLS[1:-2]
CELLS_E.append(code("run = " + repr(RUN_E) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                    'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_E.append(CELLS[-1])
(HERE / "KAGGLE_runE.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_E), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_E.sh").write_text(RUN_E, encoding="utf-8", newline="\n")
print("wrote KAGGLE_runE.ipynb", len(CELLS_E), "cells")

# Variant of run D's blend at t 0.80 with France 0.80 too (no training; reads D's saved test predictions). CPU is enough.
RUN_VD = r"""#!/bin/bash
set -e
W=/tmp/work; OUT=/kaggle/working; mkdir -p $W
CORE=$(dirname "$(find /kaggle/input -name test_records.parquet | head -1)")
for f in $CORE/test_norm2.parquet $CORE/test_records.parquet $CORE/validate_submission.py; do ln -sf $f $W/; done
for f in $(find /kaggle/input -name 'p7_v3_test_*.parquet' -o -name 'p8_v3_test_*.parquet'); do ln -sf "$f" $W/; done
echo "linked $(ls $W | wc -l) files (need 3 + 16 predictions)"; [ $(ls $W/p7_v3_test_*.parquet $W/p8_v3_test_*.parquet | wc -l) -eq 16 ] || { echo "MISSING p7/p8 test predictions: add the Output of KAGGLE_runD"; exit 1; }
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s15_blend.py write blend78ft_t80 0.80 0.80 7 8
python $W/validate_submission.py --test-dir $CORE --matching $OUT/blend78ft_t80/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend78ft_t80 $(date)"
"""
CELLS_VD = [md("""
# Run D variant: blend78ft at t 0.80 with France 0.80 (instead of 0.90)
**Add Input:** `mlc26-core` (Datasets) and the Output of `KAGGLE_runD` (Notebooks). No GPU needed. Save & Run All (~5 min).
Output: `blend78ft_t80/matching_results.tsv`. Upload it right after `blend78ft_fr90`: the LB difference / 0.1497 = France's F0.5 change.
""")] + CELLS[1:-2]
CELLS_VD.append(code("run = " + repr(RUN_VD) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                     'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_VD.append(CELLS[-1])
(HERE / "KAGGLE_variantD.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_VD), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_VD.sh").write_text(RUN_VD, encoding="utf-8", newline="\n")
print("wrote KAGGLE_variantD.ipynb", len(CELLS_VD), "cells")

# Run E variant: blend910 at t 0.80 with France 0.95 (France got better 0.80 -> 0.90 on the LB: over-confident there).
RUN_VE = (RUN_VD.replace("p7_v3_test_", "p9_v3_test_").replace("p8_v3_test_", "p10_v3_test_")
          .replace("p7/p8", "p9/p10").replace("KAGGLE_runD", "KAGGLE_runE")
          .replace("s15_blend.py write blend78ft_t80 0.80 0.80 7 8", "s15_blend.py write blend910_fr95 0.80 0.95 9 10")
          .replace("blend78ft_t80", "blend910_fr95"))
assert "p9_v3_test_" in RUN_VE and "p7" not in RUN_VE and "0.80 0.95 9 10" in RUN_VE
CELLS_VE = [md("""
# Run E variant: blend910 at t 0.80 with France 0.95 (instead of 0.90)
**Add Input:** `mlc26-core` (Datasets) and the Output of `KAGGLE_runE` (Notebooks). No GPU needed. Save & Run All (~5 min).
Output: `blend910_fr95/matching_results.tsv` (same as blend910_fr90 except France rows).
""")] + CELLS[1:-2]
CELLS_VE.append(code("run = " + repr(RUN_VE) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                     'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_VE.append(CELLS[-1])
(HERE / "KAGGLE_variantE.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_VE), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_VE.sh").write_text(RUN_VE, encoding="utf-8", newline="\n")
print("wrote KAGGLE_variantE.ipynb", len(CELLS_VE), "cells")

# Run R = Fable review 3 rules: rebuild TRAIN features of D's union, OOF p7 from D's saved models (must reproduce D's 0.98702),
# test every rule on labelled OOF (rules.json), apply the passing ones to D's test predictions (blend78ft_rules).
_TOP = RUN_D.split('echo "== cosine')[0].replace(KNNFT_LINK, E_PREFLIGHT)
_i = _TOP.index("for f in train_records.parquet"); _TOP = _TOP[:_i] + _TOP[_TOP.index(_END, _i) + len(_END):]
RUN_R = _TOP.replace("# Run D: run C union + stage 7/8", "# Run R: decode-time rules validated on train OOF") + r"""for f in $(find /kaggle/input \( -name 'xgb_s7_v3_f*.json' -o -name 'p7_v3_test_*.parquet' -o -name 'p8_v3_test_*.parquet' \)); do ln -sf "$f" $W/; done
[ -e $W/xgb_s7_v3_f0.json ] && [ -e $W/xgb_s7_v3_f1.json ] && [ $(ls $W/p7_v3_test_*.parquet $W/p8_v3_test_*.parquet | wc -l) -eq 16 ] \
  || { echo "PREFLIGHT FAILED: run D's stage-7 models / test predictions not found -> add the Output of KAGGLE_runD as input"; exit 1; }
echo "run D models + test predictions found"
python s20_rules.py selftest
for f in $(find /kaggle/input -name oof_p7_ge05.parquet); do ln -sf "$f" $W/; done
if [ -e $W/oof_p7_ge05.parquet ]; then
  echo "== OOF p7 found in the inputs (earlier run R): skipping the train feature rebuild"; cp $W/oof_p7_ge05.parquet $OUT/state/
else
  echo "== cosine (train) $(date)"; python s10_v3.py cos train
  echo "== feat3 (train) $(date)"
  for j in 0 1 2 3 4 5 6 7; do python s10_v3.py build train $j; done
  python s10_v3.py report
  echo "== address add-on + competition features (train) $(date)"
  python s14_addr.py train; python s16_idcomp.py train
  echo "== OOF from run D's stage-7 models (must reproduce D: best t=0.7 F=0.98702) $(date)"
  python s20_rules.py oof
fi
echo "== rules on train OOF $(date)"
python s20_rules.py rules
cp $W/rules.json $OUT/state/ 2>/dev/null || { mkdir -p $OUT/state; cp $W/rules.json $OUT/state/; }
python s20_rules.py apply blend78ft_rules $W/rules.json 7 8
$VAL $OUT/blend78ft_rules/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend78ft_rules $(date)"
du -sh $OUT; echo "== KAGGLE RUN R DONE $(date)"
"""
assert "s20_rules.py oof" in RUN_R and 'echo "MISSING' not in RUN_R and "s19_preflight" in RUN_R
CELLS_R = [md("""
# Run R: decode-time rules for generic-name false positives, validated on train OOF (Fable review 3)
**Add Input:** `mlc26-core`, `mlc26-emb`, `mlc26-v3` (Datasets) + Outputs of `KAGGLE_runC` and `KAGGLE_runD` (Notebooks).
GPU T4 x2. Save & Run All (~1.5 h). Look for: the OOF line (must say F=0.98702), the rule table (PASS / -), `SELECTED ...`.
Output: `state/rules.json` (for run E's predictions: KAGGLE_applyE) and `blend78ft_rules/matching_results.tsv`.
""")] + CELLS[1:-2]
CELLS_R.append(code("run = " + repr(RUN_R) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                    'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_R.append(CELLS[-1])
(HERE / "KAGGLE_runR.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_R), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_R.sh").write_text(RUN_R, encoding="utf-8", newline="\n")
print("wrote KAGGLE_runR.ipynb", len(CELLS_R), "cells")

# Apply run R's rules to run E's test predictions (CPU, ~5 min).
RUN_AE = r"""#!/bin/bash
set -e
W=/tmp/work; OUT=/kaggle/working; mkdir -p $W
CORE=$(dirname "$(find /kaggle/input -name test_records.parquet | head -1)")
for f in $CORE/test_norm2.parquet $CORE/test_norm3.parquet $CORE/test_records.parquet $CORE/validate_submission.py; do ln -sf $f $W/; done
for f in $(find /kaggle/input \( -name 'p9_v3_test_*.parquet' -o -name 'p10_v3_test_*.parquet' \)); do ln -sf "$f" $W/; done
RJ=$(find /kaggle/input -name rules.json | head -1)
[ -n "$RJ" ] && [ $(ls $W/p9_v3_test_*.parquet $W/p10_v3_test_*.parquet | wc -l) -eq 16 ] \
  || { echo "PREFLIGHT FAILED: need the Outputs of KAGGLE_runE (p9/p10 test predictions) and KAGGLE_runR (rules.json)"; exit 1; }
echo "rules: $RJ"; cat $RJ | head -3
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s20_rules.py selftest
python s20_rules.py apply blend910_rules $RJ 9 10
python $W/validate_submission.py --test-dir $CORE --matching $OUT/blend910_rules/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend910_rules $(date)"
"""
CELLS_AE = [md("""
# Apply run R's rules to run E's predictions -> blend910_rules
**Add Input:** `mlc26-core` (Datasets) + Outputs of `KAGGLE_runE` and `KAGGLE_runR` (Notebooks). No GPU. Save & Run All (~5 min).
""")] + CELLS[1:-2]
CELLS_AE.append(code("run = " + repr(RUN_AE) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                     'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_AE.append(CELLS[-1])
(HERE / "KAGGLE_applyE.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_AE), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_AE.sh").write_text(RUN_AE, encoding="utf-8", newline="\n")
print("wrote KAGGLE_applyE.ipynb", len(CELLS_AE), "cells")

# Expected-F0.5 decode of run E's test predictions (Fable review 4). CPU. Parameters come from the OOF tuning (s21 tune).
EF = dict(floor=0.6, gamma=1.0, floor_fr=0.9, gamma_fr=1.0)   # <- set from the OOF tuning before building
RUN_DE = r"""#!/bin/bash
set -e
W=/tmp/work; OUT=/kaggle/working; mkdir -p $W
CORE=$(dirname "$(find /kaggle/input -name test_records.parquet | head -1)")
for f in $CORE/test_norm2.parquet $CORE/test_records.parquet $CORE/validate_submission.py; do ln -sf $f $W/; done
for f in $(find /kaggle/input \( -name 'p9_v3_test_*.parquet' -o -name 'p10_v3_test_*.parquet' \)); do ln -sf "$f" $W/; done
[ $(ls $W/p9_v3_test_*.parquet $W/p10_v3_test_*.parquet | wc -l) -eq 16 ] || { echo "PREFLIGHT FAILED: add the Output of KAGGLE_runE"; exit 1; }
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s21_efdecode.py selftest
python s21_efdecode.py apply blend910_ef FLOOR GAMMA FLOOR_FR GAMMA_FR 9 10
python $W/validate_submission.py --test-dir $CORE --matching $OUT/blend910_ef/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend910_ef $(date)"
""".replace("FLOOR_FR", str(EF["floor_fr"])).replace("GAMMA_FR", str(EF["gamma_fr"])).replace("FLOOR", str(EF["floor"])).replace("GAMMA", str(EF["gamma"]))
CELLS_DE = [md(f"""
# Run E + per-S1 expected-F0.5 decoding -> blend910_ef   (params {EF})
**Add Input:** `mlc26-core` (Datasets) + Output of `KAGGLE_runE` (Notebooks). No GPU. Save & Run All (~5 min).
Look for `SELFTEST OK`, the per-country table (ef_per_S1 vs fr90_per_S1), `PASS`.
""")] + CELLS[1:-2]
CELLS_DE.append(code("run = " + repr(RUN_DE) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                     'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_DE.append(CELLS[-1])
(HERE / "KAGGLE_decodeE.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_DE), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_DE.sh").write_text(RUN_DE, encoding="utf-8", newline="\n")
print("wrote KAGGLE_decodeE.ipynb", len(CELLS_DE), "cells", EF)

# Last sprint: set-level stacker on saved probabilities (s22_stack.py). CPU only.
RUN_S = r"""#!/bin/bash
set -e
W=/tmp/work; OUT=/kaggle/working; mkdir -p $W
CORE=$(dirname "$(find /kaggle/input -name test_records.parquet | head -1)")
for f in train_records.parquet test_records.parquet train_gt.parquet test_norm2.parquet validate_submission.py; do ln -sf $CORE/$f $W/; done
for f in $(find /kaggle/input \( -name 'oof_p7_ge05.parquet' -o -name 'p7_v3_test_*.parquet' \)); do ln -sf "$f" $W/; done
[ -e $W/oof_p7_ge05.parquet ] && [ $(ls $W/p7_v3_test_*.parquet | wc -l) -eq 8 ] && [ -e $W/train_records.parquet ] \
  || { echo "PREFLIGHT FAILED: need mlc26-core + Output of KAGGLE_runR (oof_p7_ge05.parquet) + Output of KAGGLE_runD (p7 test predictions)"; exit 1; }
echo "PREFLIGHT OK: OOF + 8 run-D test prediction files + records found"
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s22_stack.py selftest
python s22_stack.py run stack7
python $W/validate_submission.py --test-dir $CORE --matching $OUT/stack7/matching_results.tsv | tail -1
echo "== SUBMISSION READY: stack7 $(date)"
"""
CELLS_S = [md("""
# Last sprint: set-level stacker on run D's probabilities -> stack7
**Add Input:** `mlc26-core` (Datasets) + Outputs of `KAGGLE_runR` and `KAGGLE_runD` (Notebooks). **No GPU** (CPU session). ~30 min.
Look for: `PREFLIGHT OK`, `SELFTEST OK`, the `OOF ... delta` line (laptop: +0.00078), the per-country table, `PASS`.
""")] + CELLS[1:-2]
CELLS_S.append(code("run = " + repr(RUN_S) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                    'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_S.append(CELLS[-1])
(HERE / "KAGGLE_stack.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_S), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_S.sh").write_text(RUN_S, encoding="utf-8", newline="\n")
print("wrote KAGGLE_stack.ipynb", len(CELLS_S), "cells")

# Run X (high risk / high reward): fold-wise cross-encoder on the uncertain band + set-level stacker. GPU T4 x2, ~1.3 h.
RUN_X = r"""#!/bin/bash
set -e
W=/tmp/work; OUT=/kaggle/working; mkdir -p $W $OUT/state/ceband
CORE=$(dirname "$(find /kaggle/input -name test_records.parquet | head -1)")
for f in train_records.parquet test_records.parquet train_gt.parquet test_norm2.parquet validate_submission.py; do ln -sf $CORE/$f $W/; done
for f in $(find /kaggle/input \( -name 'oof_p7_ge05.parquet' -o -name 'p7_v3_test_*.parquet' -o -name 'ceband_*.parquet' \)); do ln -sf "$f" $W/; done
for d in $(find /kaggle/input -type d -name 'ceband_model_f*'); do ln -sfn "$d" $W/; done
CE=$(find /kaggle/input -type d -name 'mmarco-mMiniLMv2-L12-H384-v1' | head -1)
if [ -z "$CE" ]; then
  echo "cross-encoder base not in the inputs: downloading cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0)"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', local_dir='/tmp/mmarco', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])" \
    || { echo "PREFLIGHT FAILED: no internet for the model download -> Session options -> Internet On"; exit 1; }
  CE=/tmp/mmarco
fi
NG=$(python -c "import torch; print(torch.cuda.device_count())")
[ -e $W/oof_p7_ge05.parquet ] && [ $(ls $W/p7_v3_test_*.parquet | wc -l) -eq 8 ] && [ -e $W/train_records.parquet ] && [ "$NG" -ge 2 ] && [ -e $CE/config.json ] \
  || { echo "PREFLIGHT FAILED: need mlc26-core + Outputs of KAGGLE_runR and KAGGLE_runD + GPU T4 x2 (GPUs seen: $NG) + CE base ($CE)"; exit 1; }
echo "PREFLIGHT OK: OOF + 8 run-D test files + records | GPUs $NG | CE base $CE"
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3 BER_CE_MODEL=$CE BER_STACK_DEVICE=cuda
python s22_stack.py selftest
echo "== cross-encoder: train the two fold models on the uncertain OOF band, one per GPU $(date)"
CUDA_VISIBLE_DEVICES=0 python s23_ceband.py train 0 > $OUT/ceX_f0.log 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=1 python s23_ceband.py train 1 > $OUT/ceX_f1.log 2>&1 & P2=$!
wait $P1 || { tail -30 $OUT/ceX_f0.log; exit 1; }
wait $P2 || { tail -30 $OUT/ceX_f1.log; exit 1; }
grep -h "fold\|pairs/s" $OUT/ceX_f0.log $OUT/ceX_f1.log | tail -6
cp -r $W/ceband_model_f0 $W/ceband_model_f1 $OUT/state/ 2>/dev/null || true
echo "== cross-encoder: score OOF band (other-fold model) + test band (both) $(date)"
python s23_ceband.py score
cp $W/ceband_*.parquet $OUT/state/ceband/
echo "== stacker WITHOUT the cross-encoder (reference) $(date)"
python s22_stack.py run stack7
$VAL_CMD
echo "== stacker WITH the cross-encoder $(date)"
python $W/validate_submission.py --test-dir $CORE --matching $OUT/stack7/matching_results.tsv | tail -1
BER_STACK_CE=1 python s22_stack.py run stack7ce || echo "== stack7ce NOT better on OOF -> not written (use stack7)"
[ -e $OUT/stack7ce/matching_results.tsv ] && python $W/validate_submission.py --test-dir $CORE --matching $OUT/stack7ce/matching_results.tsv | tail -1
echo "== SUBMISSION READY: stack7 and stack7ce (upload the one with the higher 'stacked' OOF line) $(date)"
""".replace("$VAL_CMD\n", "")
CELLS_X = [md("""
# Run X: fold-wise cross-encoder on the uncertain pairs + set-level stacker -> stack7 / stack7ce
**Add Input:** `mlc26-core` (Datasets) + Outputs of `KAGGLE_runR` and `KAGGLE_runD` (Notebooks). **GPU T4 x2, Internet On.** ~1.3 h.
Look for: `PREFLIGHT OK`, `SELFTEST OK`, the two `OOF (US+India) ... stacked ...` lines (without / with the cross-encoder), `PASS` x2.
""")] + CELLS[1:-2]
CELLS_X.append(code("run = " + repr(RUN_X) + '\nopen("/kaggle/working/run_kaggle.sh", "w").write(run)\n'
                    'print(subprocess.run("bash -n /kaggle/working/run_kaggle.sh && echo syntax-ok", shell=True, capture_output=True, text=True).stdout)'))
CELLS_X.append(CELLS[-1])
(HERE / "KAGGLE_runX.ipynb").write_text(json.dumps(dict(nb, cells=CELLS_X), indent=1, ensure_ascii=False), encoding="utf-8")
(HERE / "run_kaggle_X.sh").write_text(RUN_X, encoding="utf-8", newline="\n")
print("wrote KAGGLE_runX.ipynb", len(CELLS_X), "cells")
