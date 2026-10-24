#!/bin/bash
set -e
D="D:/Amazzon/smoke_v3/drive"
cd D:/Amazzon/smoke_code/x/src
export BER_WORK=D:/Amazzon/smoke_v3/workB BER_DATA="$D/student_resource/dataset"
SYNC="$D/work_v3"; mkdir -p "$SYNC"
VAL="$D/student_resource/utils/validate_submission.py"; TEST="$D/student_resource/dataset/test"
save() { # copy to Drive under a temp name, rename only when complete: a session dying mid-copy never leaves a truncated file
  for f in "$@"; do [ -e "$f" ] || continue; b=$(basename "$f")
    [ "$(du -sb "$f" | cut -f1)" = "$(du -sb "$SYNC/$b" 2>/dev/null | cut -f1)" ] && continue
    rm -rf "$SYNC/$b.part"
    if cp -r "$f" "$SYNC/$b.part"; then rm -rf "$SYNC/$b"; mv "$SYNC/$b.part" "$SYNC/$b"; else echo "WARN: Drive copy failed for $b"; fi
  done; }
submit() { # stage outdir name
  BER_OUT=$2 python s6_write.py $1
  python "$VAL" --matching $2/matching_results.tsv --candidate $2/candidate_pairs.tsv --test-dir "$TEST"
  mkdir -p "$D/$3"; cp $2/*.tsv "$D/$3/"; echo "== SUBMISSION READY: MLC'26/$3 $(date)"; }
[ $(ls D:/Amazzon/smoke_v3/workB | grep -c knn_) = 16 ] || { echo "ERROR: need 16 knn files (train + test) in D:/Amazzon/smoke_v3/workB"; exit 1; }
nfeat() { ls D:/Amazzon/smoke_v3/workB | grep -c "^$1_feat3_[0-7].parquet$"; }
if [ "${BER_V3_MODE:-full}" = "test" ]; then   # helper account: build only the test half, then stop
  echo "== TEST-HALF MODE: cosine + feat3 for test $(date)"
  python s10_v3.py cos test
  for j in 0 1 2 3 4 5 6 7; do cp -n "$SYNC/test_feat3_$j.parquet" D:/Amazzon/smoke_v3/workB/ 2>/dev/null || true
    python s10_v3.py build test $j; save D:/Amazzon/smoke_v3/workB/test_feat3_$j.parquet; done
  echo "== TEST FEAT3 DONE $(date)"; exit 0
fi
echo "== train: cosine + feat3 $(date)"
[ $(nfeat train) = 8 ] || python s10_v3.py cos train
for j in 0 1 2 3 4 5 6 7; do python s10_v3.py build train $j; done   # not saved to Drive (8 GB; rebuilt in ~70 min if needed)
echo "== test: take buckets the helper account already saved, build the rest $(date)"
cp -n "$SYNC"/test_feat3_*.parquet D:/Amazzon/smoke_v3/workB/ 2>/dev/null || true
[ $(nfeat test) = 8 ] || python s10_v3.py cos test
for j in 0 1 2 3 4 5 6 7; do cp -n "$SYNC/test_feat3_$j.parquet" D:/Amazzon/smoke_v3/workB/ 2>/dev/null || true
  python s10_v3.py build test $j; done   # only the helper account saves test feat3 (quota)
python s10_v3.py report
export BER_FEAT=feat3 BER_TAG=_v3
echo "== stage1 $(date)"; python s4_train.py 1
save D:/Amazzon/smoke_v3/workB/xgb_s1_v3_* D:/Amazzon/smoke_v3/workB/val_stage1_v3.json D:/Amazzon/smoke_v3/workB/p1_v3_*.parquet
[ -e "$D/output_v3s1/matching_results.tsv" ] || submit 1 D:/Amazzon/smoke_v3/out_v3s1 output_v3s1
echo "== cross-encoder $(date)"; python s8_ce.py all
save D:/Amazzon/smoke_v3/workB/ce_model_f0 D:/Amazzon/smoke_v3/workB/ce_model_f1 D:/Amazzon/smoke_v3/workB/ce_v3_*.parquet
echo "== stage4 (stage 1 + cross-encoder, no stacking) $(date)"; python s4_train.py 4
save D:/Amazzon/smoke_v3/workB/xgb_s4_v3_* D:/Amazzon/smoke_v3/workB/val_stage4_v3.json
[ -e "$D/output_v3s4/matching_results.tsv" ] || submit 4 D:/Amazzon/smoke_v3/out_v3s4 output_v3s4
echo "== comp (fold-consistent) $(date)"; python s5_comp.py
echo "== stage2 $(date)"; python s4_train.py 2
save D:/Amazzon/smoke_v3/workB/xgb_s2_v3_* D:/Amazzon/smoke_v3/workB/val_stage2_v3.json
[ -e "$D/output_v3s2/matching_results.tsv" ] || submit 2 D:/Amazzon/smoke_v3/out_v3s2 output_v3s2
echo "== stage3 $(date)"; python s4_train.py 3
save D:/Amazzon/smoke_v3/workB/xgb_s3_v3_* D:/Amazzon/smoke_v3/workB/val_stage3_v3.json
[ -e "$D/output_v3s3/matching_results.tsv" ] || submit 3 D:/Amazzon/smoke_v3/out_v3s3 output_v3s3
echo "== V3 ALL DONE $(date)"
