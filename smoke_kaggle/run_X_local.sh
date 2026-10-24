#!/bin/bash
set -e
W=D:/Amazzon/smoke_kaggle/workX; OUT=D:/Amazzon/smoke_kaggle/workingX; mkdir -p $W $OUT/state/ceband
CORE=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name test_records.parquet | head -1)")
for f in train_records.parquet test_records.parquet train_gt.parquet test_norm2.parquet validate_submission.py; do ln -sf $CORE/$f $W/; done
for f in $(find D:/Amazzon/smoke_kaggle/workingR/state D:/Amazzon/smoke_kaggle/workingD/state \( -name 'oof_p7_ge05.parquet' -o -name 'p7_v3_test_*.parquet' -o -name 'ceband_*.parquet' \)); do ln -sf "$f" $W/; done
for d in $(find D:/Amazzon/smoke_kaggle/input_ce -type d -name 'ceband_model_f*'); do ln -sfn "$d" $W/; done
CE=$(find D:/Amazzon/smoke_kaggle/input_ce -type d -name 'mmarco-mMiniLMv2-L12-H384-v1' | head -1)
if [ -z "$CE" ]; then
  echo "cross-encoder base not in the inputs: downloading cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0)"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', local_dir='/tmp/mmarco', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])" \
    || { echo "PREFLIGHT FAILED: no internet for the model download -> Session options -> Internet On"; exit 1; }
  CE=/tmp/mmarco
fi
NG=$(python -c "import torch; print(torch.cuda.device_count())")
[ -e $W/oof_p7_ge05.parquet ] && [ $(ls $W/p7_v3_test_*.parquet | wc -l) -eq 8 ] && [ -e $W/train_records.parquet ] && [ "$NG" -ge 1 ] && [ -e $CE/config.json ] \
  || { echo "PREFLIGHT FAILED: need mlc26-core + Outputs of KAGGLE_runR and KAGGLE_runD + GPU T4 x2 (GPUs seen: $NG) + CE base ($CE)"; exit 1; }
echo "PREFLIGHT OK: OOF + 8 run-D test files + records | GPUs $NG | CE base $CE"
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3 BER_CE_MODEL=$CE BER_STACK_DEVICE=cuda
python s22_stack.py selftest
echo "== cross-encoder: train the two fold models on the uncertain OOF band, one per GPU $(date)"
CUDA_VISIBLE_DEVICES=0 python s23_ceband.py train 0 > $OUT/ceX_f0.log 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=0 python s23_ceband.py train 1 > $OUT/ceX_f1.log 2>&1 & P2=$!
wait $P1 || { tail -30 $OUT/ceX_f0.log; exit 1; }
wait $P2 || { tail -30 $OUT/ceX_f1.log; exit 1; }
grep -h "fold\|pairs/s" $OUT/ceX_f0.log $OUT/ceX_f1.log | tail -6
cp -r $W/ceband_model_f0 $W/ceband_model_f1 $OUT/state/ 2>/dev/null || true
echo "== cross-encoder: score OOF band (other-fold model) + test band (both) $(date)"
python s23_ceband.py score
cp $W/ceband_*.parquet $OUT/state/ceband/
echo "== stacker WITHOUT the cross-encoder (reference) $(date)"
python s22_stack.py run stack7
echo "== stacker WITH the cross-encoder $(date)"
BER_STACK_CE=1 python s22_stack.py run stack7ce
python $W/validate_submission.py --test-dir $CORE --matching $OUT/stack7/matching_results.tsv | tail -1
python $W/validate_submission.py --test-dir $CORE --matching $OUT/stack7ce/matching_results.tsv | tail -1
echo "== SUBMISSION READY: stack7 and stack7ce (upload the one with the higher 'stacked' OOF line) $(date)"
