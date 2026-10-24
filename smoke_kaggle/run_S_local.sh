#!/bin/bash
set -e
W=D:/Amazzon/smoke_kaggle/workS; OUT=D:/Amazzon/smoke_kaggle/workingS; mkdir -p $W
CORE=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name test_records.parquet | head -1)")
for f in train_records.parquet test_records.parquet train_gt.parquet test_norm2.parquet validate_submission.py; do ln -sf $CORE/$f $W/; done
for f in $(find D:/Amazzon/smoke_kaggle/workingR/state D:/Amazzon/smoke_kaggle/workingD/state \( -name 'oof_p7_ge05.parquet' -o -name 'p7_v3_test_*.parquet' \)); do ln -sf "$f" $W/; done
[ -e $W/oof_p7_ge05.parquet ] && [ $(ls $W/p7_v3_test_*.parquet | wc -l) -eq 8 ] && [ -e $W/train_records.parquet ] \
  || { echo "PREFLIGHT FAILED: need mlc26-core + Output of KAGGLE_runR (oof_p7_ge05.parquet) + Output of KAGGLE_runD (p7 test predictions)"; exit 1; }
echo "PREFLIGHT OK: OOF + 8 run-D test prediction files + records found"
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s22_stack.py selftest
python s22_stack.py run stack7
python $W/validate_submission.py --test-dir $CORE --matching $OUT/stack7/matching_results.tsv | tail -1
echo "== SUBMISSION READY: stack7 $(date)"
