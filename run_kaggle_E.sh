#!/bin/bash
# Run E: run D + fold-wise fine-tuned e5 cosine features (stage 9/10) (+ a second, differently-configured stage-5 model and their blend) on Kaggle (31 GB RAM, 4 CPUs, T4).
set -e
W=/tmp/work; OUT=/kaggle/working; mkdir -p $W $OUT/state
CORE=$(dirname "$(find /kaggle/input -name train_records.parquet | head -1)")
EMB=$(dirname "$(find /kaggle/input -name train_emb.f16 | head -1)")
V3=$(dirname "$(find /kaggle/input -name ce_v3_train_0.parquet | head -1)")
echo "inputs: core=$CORE emb=$EMB v3=$V3"
for d in "$CORE" "$EMB" "$V3"; do for f in "$d"/*; do case "$f" in *addr4_*) continue;; esac; [ -f "$f" ] && ln -sf "$f" $W/; done; done
for m in ce_model_f0 ce_model_f1; do d=$(find /kaggle/input -type d -name $m | head -1); [ -n "$d" ] && ln -sfn "$d" $W/$m || true; done   # not needed by stage 5/6
echo "linked $(ls $W | wc -l) inputs"
cd $OUT/src
export BER_WORK=$W BER_OUT=$OUT BER_FEAT=feat3 BER_TAG=_v3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BER_NPIECE=8
free -g | head -2
VAL="python $W/validate_submission.py --test-dir $CORE --matching"
for f in $(find /kaggle/input -name 'knnft_*.parquet'); do ln -sf "$f" $W/; done
E5=$(dirname "$(find /kaggle/input -name model.safetensors -path '*multilingual-e5-small*' | grep -v -e ftmodel -e e5_ft | head -1)")
echo "== PREFLIGHT: inputs, GPUs, e5, disk $(date)"
E5="$E5" python s19_preflight.py
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
echo "== cosine $(date)"; python s10_v3.py cos train; python s10_v3.py cos test
echo "== feat3 $(date)"
for s in train test; do for j in 0 1 2 3 4 5 6 7; do python s10_v3.py build $s $j; done; done
python s10_v3.py report
echo "== address add-on for the new union $(date)"
python s14_addr.py train; python s14_addr.py test
echo "== per-record competition features $(date)"
python s16_idcomp.py train; python s16_idcomp.py test
echo "== stage9 = stage 7 + fine-tuned cosine features, FULL DATA $(date)"
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
