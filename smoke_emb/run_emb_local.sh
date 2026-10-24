#!/bin/bash
set -e
D="D:/Amazzon/smoke_emb/drive"
cd D:/Amazzon/smoke_code/x/src
OUT="$D/work_emb"
export BER_WORK=D:/Amazzon/smoke_emb/work BER_EMB_MODEL="$D/models/multilingual-e5-small" BER_EMB_SYNC="$OUT"
echo "== encode test $(date)"
python e1_emb.py encode test
[ "$(stat -c%s "$OUT/test_emb.f16" 2>/dev/null)" = "$(stat -c%s D:/Amazzon/smoke_emb/work/test_emb.f16)" ] || cp D:/Amazzon/smoke_emb/work/test_emb.f16 D:/Amazzon/smoke_emb/work/test_emb_ids.npy "$OUT/"
[ "$(stat -c%s "$OUT/test_emb.f16")" = "$(stat -c%s D:/Amazzon/smoke_emb/work/test_emb.f16)" ] && rm -f "$OUT"/test_emb_c*.f16   # full file safe on Drive -> drop the chunks
echo "== knn test $(date)"
python e1_emb.py knn test
cp D:/Amazzon/smoke_emb/work/knn_test_*.parquet "$OUT/"
echo "== EMB TEST DONE $(date)"
