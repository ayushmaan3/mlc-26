#!/bin/bash
set -e
W=D:/Amazzon/smoke_kaggle/workVE; OUT=D:/Amazzon/smoke_kaggle/workingVE; mkdir -p $W
CORE=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name test_records.parquet | head -1)")
for f in $CORE/test_norm2.parquet $CORE/test_records.parquet $CORE/validate_submission.py; do ln -sf $f $W/; done
for f in $(find D:/Amazzon/smoke_kaggle/workingE/state/preds -name 'p9_v3_test_*.parquet' -o -name 'p10_v3_test_*.parquet'); do ln -sf "$f" $W/; done
echo "linked $(ls $W | wc -l) files (need 3 + 16 predictions)"; [ $(ls $W/p9_v3_test_*.parquet $W/p10_v3_test_*.parquet | wc -l) -eq 16 ] || { echo "MISSING p9/p10 test predictions: add the Output of KAGGLE_runE"; exit 1; }
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s15_blend.py write blend910_fr95 0.80 0.95 9 10
python $W/validate_submission.py --test-dir $CORE --matching $OUT/blend910_fr95/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend910_fr95 $(date)"
