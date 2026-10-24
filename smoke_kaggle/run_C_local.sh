#!/bin/bash
# Run C: + candidates from the fine-tuned e5 (s17), then full-data stage 5 (+ a second, differently-configured stage-5 model and their blend) on Kaggle (31 GB RAM, 4 CPUs, T4).
set -e
W=D:/Amazzon/smoke_kaggle/workC; OUT=D:/Amazzon/smoke_kaggle/workingC; mkdir -p $W $OUT/state
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
FT=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name model.safetensors -path '*ftmodel*' | head -1)")
echo "fine-tuned e5: $FT"
for f in $(find D:/Amazzon/smoke_kaggle/input -name 'knnft_*.parquet'); do ln -sf "$f" $W/; done
[ -e "$FT/model.safetensors" ] || [ -e $W/knnft_test_7.parquet ] || { echo "MISSING fine-tuned model: add the mlc26-ftmodel dataset"; exit 1; }
if [ ! -e $W/knnft_test_7.parquet ]; then
  echo "== encode train + test with the fine-tuned e5, one per GPU $(date)"
  F=D:/Amazzon/smoke_kaggle/ftC; mkdir -p $F; ln -sf $W/train_records.parquet $W/test_records.parquet $F/
  BER_WORK=$F BER_EMB_MODEL=$FT CUDA_VISIBLE_DEVICES=0 python e1_emb.py encode train > $OUT/ft_enc_train.log 2>&1 & P1=$!
  BER_WORK=$F BER_EMB_MODEL=$FT CUDA_VISIBLE_DEVICES=0 python e1_emb.py encode test > $OUT/ft_enc_test.log 2>&1 & P2=$!
  wait $P1; wait $P2; tail -n 2 $OUT/ft_enc_train.log $OUT/ft_enc_test.log; rm -f $F/*_emb_c*.f16
  echo "== kNN@20 with the fine-tuned e5 $(date)"
  BER_WORK=$F CUDA_VISIBLE_DEVICES=0 python e1_emb.py knn train > $OUT/ft_knn_train.log 2>&1 & P1=$!
  BER_WORK=$F CUDA_VISIBLE_DEVICES=0 python e1_emb.py knn test > $OUT/ft_knn_test.log 2>&1 & P2=$!
  wait $P1; wait $P2
  mkdir -p $OUT/state/knnft
  for s in train test; do for j in 0 1 2 3 4 5 6 7; do
    cp $F/knn_${s}_${j}.parquet $OUT/state/knnft/knnft_${s}_${j}.parquet; ln -sf $OUT/state/knnft/knnft_${s}_${j}.parquet $W/; done; done
  rm -rf $F
fi
echo "== cosine $(date)"; python s10_v3.py cos train; python s10_v3.py cos test
echo "== feat3 $(date)"
for s in train test; do for j in 0 1 2 3 4 5 6 7; do python s10_v3.py build $s $j; done; done
python s10_v3.py report
echo "== address add-on for the new union $(date)"
python s14_addr.py train; python s14_addr.py test
echo "== stage5 FULL DATA $(date)"
python s4_train.py 5 || { echo "== stage5 full data failed (likely RAM): retrying the unfinished fold(s) on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 5; }
cp $W/val_stage5_v3.json $W/xgb_s5_v3_* $OUT/state/
python s15_blend.py v3s5ft 5
$VAL $OUT/v3s5ft/matching_results.tsv | tail -1; $VAL $OUT/v3s5ft_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: v3s5ft + v3s5ft_fr90 $(date)"
echo "== stage6: second stage-5 model (511 leaves, other seed/columns) $(date)"
export BER_LEAVES=511 BER_SEED=1 BER_COLS=0.7 BER_MCW=40
python s4_train.py 6 || { echo "== stage6 full data failed: retrying on half data"; BER_TRAIN_BUCKETS=2 python s4_train.py 6; }
cp $W/val_stage6_v3.json $W/xgb_s6_v3_* $OUT/state/
python s15_blend.py blend56ft 5 6
$VAL $OUT/blend56ft/matching_results.tsv | tail -1; $VAL $OUT/blend56ft_fr90/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend56ft + blend56ft_fr90 $(date)"
mkdir -p $OUT/state/preds; cp $W/p5_v3_test_*.parquet $W/p6_v3_test_*.parquet $W/val_blend_*.json $OUT/state/preds/ 2>/dev/null || true
du -sh $OUT; echo "== KAGGLE RUN C DONE $(date)"
