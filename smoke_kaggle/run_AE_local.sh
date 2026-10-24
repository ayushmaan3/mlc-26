#!/bin/bash
set -e
W=D:/Amazzon/smoke_kaggle/workAE; OUT=D:/Amazzon/smoke_kaggle/workingAE; mkdir -p $W
CORE=$(dirname "$(find D:/Amazzon/smoke_kaggle/input -name test_records.parquet | head -1)")
for f in $CORE/test_norm2.parquet $CORE/test_norm3.parquet $CORE/test_records.parquet $CORE/validate_submission.py; do ln -sf $f $W/; done
for f in $(find D:/Amazzon/smoke_kaggle/workingE/state/preds \( -name 'p9_v3_test_*.parquet' -o -name 'p10_v3_test_*.parquet' \)); do ln -sf "$f" $W/; done
RJ=$(find D:/Amazzon/smoke_kaggle/workingR/state -name rules.json | head -1)
[ -n "$RJ" ] && [ $(ls $W/p9_v3_test_*.parquet $W/p10_v3_test_*.parquet | wc -l) -eq 16 ] \
  || { echo "PREFLIGHT FAILED: need the Outputs of KAGGLE_runE (p9/p10 test predictions) and KAGGLE_runR (rules.json)"; exit 1; }
echo "rules: $RJ"; cat $RJ | head -3
cd $OUT/src; export BER_WORK=$W BER_OUT=$OUT BER_TAG=_v3
python s20_rules.py selftest
python s20_rules.py apply blend910_rules $RJ 9 10
python $W/validate_submission.py --test-dir $CORE --matching $OUT/blend910_rules/matching_results.tsv | tail -1
echo "== SUBMISSION READY: blend910_rules $(date)"
