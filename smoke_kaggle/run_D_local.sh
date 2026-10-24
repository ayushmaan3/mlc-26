#!/bin/bash
# Run D: run C union + stage 7/8 (+ a second, differently-configured stage-5 model and their blend) on Kaggle (31 GB RAM, 4 CPUs, T4).
set -e
W=D:/Amazzon/smoke_kaggle/workD; OUT=D:/Amazzon/smoke_kaggle/workingD; mkdir -p $W $OUT/state
CORE=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name train_records.parquet | head -1)")
EMB=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name train_emb.f16 | head -1)")
V3=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name ce_v3_train_0.parquet | head -1)")
echo "inputs: core=$CORE emb=$EMB v3=$V3"
for d in "$CORE" "$EMB" "$V3"; do for f in "$d"/*; do case "$f" in *addr4_*) continue;; esac; [ -f "$f" ] && ln -sf "$f" $W/; done; done
for m in ce_model_f0 ce_model_f1; do d=$(find D:/Amazzon/smoke_kaggle/input -type d -name $m | head -1); [ -n "$d" ] && ln -sfn "$d" $W/$m || true; done   # not needed by stage 5/6
echo "linked $(ls $W | wc -l) inputs"
for f in train_records.parquet test_records.parquet train_gt.parquet train_norm2.parquet test_norm3.parquet train_feat2_0.parquet \
         train_emb.f16 test_emb.f16 knn_test_7.parquet ce_v3_test_7.parquet test_source1.tsv validate_submission.py; do
  [ -e $W/$f ] || { echo "MISSING $f"; exit 1; }; done
cd $OUT/src
export BER_WORK=$W BER_OUT=$OUT BER_FEAT=feat3 BER_TAG=_v3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BER_NPIECE=8
free -g | head -2
VAL="python $W/validate_submission.py --test-dir $CORE --matching"
for f in $(find D:/Amazzon/smoke_kaggle/workingC/state/knnft -name 'knnft_*.parquet'); do ln -sf "$f" $W/; done
ls $W/knnft_train_{0..7}.parquet $W/knnft_test_{0..7}.parquet > /dev/null || { echo "MISSING knnft_*: add the Output of KAGGLE_runC as input"; exit 1; }
echo "reusing fine-tuned kNN of run C: $(ls $W/knnft_*.parquet | wc -l) files"
echo "== cosine $(date)"; python s10_v3.py cos train; python s10_v3.py cos test
echo "== feat3 $(date)"
for s in train test; do for j in 0 1 2 3 4 5 6 7; do python s10_v3.py build $s $j; done; done
python s10_v3.py report
echo "== address add-on for the new union $(date)"
python s14_addr.py train; python s14_addr.py test
echo "== per-record competition features $(date)"
python s16_idcomp.py train; python s16_idcomp.py test
echo "== stage7 = stage 5 + competition features, FULL DATA $(date)"
python s4_train.py 7 || { echo "== stage7 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 7; }
cp $W/val_stage7_v3.json $W/xgb_s7_v3_* $OUT/state/
python s15_blend.py v3s7ft 7
$VAL $OUT/v3s7ft/matching_results.tsv | tail -1; $VAL $OUT/v3s7ft_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: v3s7ft + v3s7ft_fr90 $(date)"
echo "== stage8: second model with competition features (511 leaves, other seed/columns) $(date)"
export BER_LEAVES=511 BER_SEED=1 BER_COLS=0.7 BER_MCW=40
python s4_train.py 8 || { echo "== stage8 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 8; }
python s15_blend.py blend78ft 7 8
$VAL $OUT/blend78ft/matching_results.tsv | tail -1; $VAL $OUT/blend78ft_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend78ft + blend78ft_fr90 $(date)"
mkdir -p $OUT/state/preds; cp $W/p7_v3_test_*.parquet $W/p8_v3_test_*.parquet $W/val_blend_*.json $OUT/state/preds/ 2>/dev/null || true
du -sh $OUT; echo "== KAGGLE RUN D DONE $(date)"
