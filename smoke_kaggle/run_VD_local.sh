#!/bin/bash
set -e
W=D:/Amazzon/smoke_kaggle/workVD; OUT=D:/Amazzon/smoke_kaggle/workingVD; mkdir -p $W
CORE=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name test_records.parquet | head -1)")
for f in $CORE/test_norm2.parquet $CORE/test_records.parquet $CORE/validate_submission.py; do ln -sf $f $W/; done
for f in $(find D:/Amazzon/smoke_kaggle/workingD/state/preds -name 'p7_v3_test_*.parquet' -o -name 'p8_v3_test_*.parquet'); do ln -sf "$f" $W/; done
echo "linked $(ls $W | wc -l) files (need 3 + 16 predictions)"; [ $(ls $W/p7_v3_test_*.parquet $W/p8_v3_test_*.parquet | wc -l) -eq 16 ] || { echo "MISSING p7/p8 test predictions: add the Output of KAGGLE_runD"; exit 1; }
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s15_blend.py write blend78ft_t80 0.80 0.80 7 8
python $W/validate_submission.py --test-dir $CORE --matching $OUT/blend78ft_t80/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend78ft_t80 $(date)"
