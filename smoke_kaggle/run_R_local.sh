#!/bin/bash
# Run R: decode-time rules validated on train OOF (+ a second, differently-configured stage-5 model and their blend) on Kaggle (31 GB RAM, 4 CPUs, T4).
set -e
W=D:/Amazzon/smoke_kaggle/workR; OUT=D:/Amazzon/smoke_kaggle/workingR; mkdir -p $W $OUT/state
CORE=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name train_records.parquet | head -1)")
EMB=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name train_emb.f16 | head -1)")
V3=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name ce_v3_train_0.parquet | head -1)")
echo "inputs: core=$CORE emb=$EMB v3=$V3"
for d in "$CORE" "$EMB" "$V3"; do for f in "$d"/*; do case "$f" in *addr4_*) continue;; esac; [ -f "$f" ] && ln -sf "$f" $W/; done; done
for m in ce_model_f0 ce_model_f1; do d=$(find D:/Amazzon/smoke_kaggle/input -type d -name $m | head -1); [ -n "$d" ] && ln -sfn "$d" $W/$m || true; done   # not needed by stage 5/6
echo "linked $(ls $W | wc -l) inputs"
cd $OUT/src
export BER_WORK=$W BER_OUT=$OUT BER_FEAT=feat3 BER_TAG=_v3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BER_NPIECE=8
free -g | head -2
VAL="python $W/validate_submission.py --test-dir $CORE --matching"
for f in $(find D:/Amazzon/smoke_kaggle/workingC/state/knnft -name 'knnft_*.parquet'); do ln -sf "$f" $W/; done
E5=$(dirname "$(find D:/Amazzon/smoke_kaggle/input_e5 -name model.safetensors -path '*multilingual-e5-small*' | grep -v -e ftmodel -e e5_ft | head -1)")
echo "== PREFLIGHT: inputs, GPUs, e5, disk $(date)"
E5="$E5" python s19_preflight.py
for f in $(find D:/Amazzon/smoke_kaggle/workingD/state \( -name 'xgb_s7_v3_f*.json' -o -name 'p7_v3_test_*.parquet' -o -name 'p8_v3_test_*.parquet' \)); do ln -sf "$f" $W/; done
[ -e $W/xgb_s7_v3_f0.json ] && [ -e $W/xgb_s7_v3_f1.json ] && [ $(ls $W/p7_v3_test_*.parquet $W/p8_v3_test_*.parquet | wc -l) -eq 16 ] \
  || { echo "PREFLIGHT FAILED: run D's stage-7 models / test predictions not found -> add the Output of KAGGLE_runD as input"; exit 1; }
echo "run D models + test predictions found"
python s20_rules.py selftest
for f in $(find D:/Amazzon/smoke_kaggle/input_R -name oof_p7_ge05.parquet); do ln -sf "$f" $W/; done
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
