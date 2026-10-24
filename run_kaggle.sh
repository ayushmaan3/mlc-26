#!/bin/bash
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
