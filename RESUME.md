# Resume on a fresh Colab T4 runtime

The pipeline can be resumed at any point: every stage skips outputs that already exist in `/content/work`.
Run the cells in order. **Cell 1 needs one Drive-auth click. After that, nothing heavy runs inside the kernel.**
Everything heavy runs as background shell processes, so a kernel crash can't take the job down with it.

## Cell 1 — mount Drive, install packages
```python
from google.colab import drive
drive.mount('/content/drive')
import subprocess
print(subprocess.run("pip install -q rapidfuzz anyascii && nvidia-smi --query-gpu=name,memory.total --format=csv && free -g && df -h /content | tail -1",
                     shell=True, capture_output=True, text=True).stdout)
```

## Cell 2 — restore data + code (shell `cp`, about 10 min)
```python
import subprocess
r = subprocess.run(r'''
D="/content/drive/MyDrive/MLC'26"
mkdir -p /content/work /content/out /content/ber/src
# base files (v1 stage outputs we still need)
cp "$D"/work/train_records.parquet "$D"/work/test_records.parquet "$D"/work/train_gt.parquet /content/work/
# every v2 artifact (norm2/norm3, idf, cand2, finished feat2 buckets, models, p*, comp*, val_stage*.json)
cp "$D"/work_v2/* /content/work/
# code
cp "$D"/code_v2/*.py /content/ber/src/ && cp "$D"/code_v2/run_rest.sh /content/run_rest.sh
rm -f /content/work/_tmp_*          # half-written sub-chunks from an interrupted bucket
ls /content/work | wc -l; du -sh /content/work; ls /content/work | grep feat2
''', shell=True, capture_output=True, text=True, executable="/bin/bash")
print(r.stdout, r.stderr[-2000:])
```

## Cell 3 — launch the driver in the background
```python
import subprocess
subprocess.Popen("nohup bash /content/run_rest.sh > /content/log_rest.txt 2>&1 &", shell=True)
```
`run_rest.sh` sets `BER_WORK=/content/work BER_OUT=/content/out BER_DATA=<Drive>/student_resource/dataset` itself.
Order: features (prep + 8 buckets x 2 splits) → stage 1 train/OOF/test → comp → stage 2 → write TSVs → validator → copy to Drive.

## Cell 4 — status check (re-run any time; cheap)
```python
import subprocess
print(subprocess.run("date; tail -15 /content/log_rest.txt; echo; ls /content/work | grep -E 'feat2|xgb|val_stage|^p[12]_|comp_' | tr '\n' ' '; echo; "
                     "pgrep -af 'run_rest|s3_features|s4_train|s5_comp|s6_write'; free -m | head -2; nvidia-smi --query-gpu=memory.used --format=csv",
                     shell=True, capture_output=True, text=True).stdout)
```

## Stop cleanly (kill by PID only, never `pkill python`)
```python
import subprocess
print(subprocess.run("pgrep -f run_rest.sh | xargs -r kill; sleep 1; pgrep -f 's3_features|s4_train|s5_comp|s6_write' | xargs -r kill; sleep 2; rm -f /content/work/_tmp_*; pgrep -af python",
                     shell=True, capture_output=True, text=True).stdout)
```

## Save state to Drive before the session ends
```python
import subprocess
print(subprocess.run(r'''D="/content/drive/MyDrive/MLC'26"; cd /content/work
for f in *_norm3.parquet *_idf_*.parquet *_feat2_*.parquet xgb_* p1_* p2_* comp_* val_stage*.json; do
  [ -e "$f" ] || continue
  if [ ! -e "$D/work_v2/$f" ] || [ $(stat -c%s "$f") != $(stat -c%s "$D/work_v2/$f") ]; then cp "$f" "$D/work_v2/"; echo "copied $f"; fi
done
cp /content/ber/src/*.py /content/run_rest.sh "$D/code_v2/"''', shell=True, capture_output=True, text=True, executable="/bin/bash").stdout)
```

## Memory rules (12.7 GB cgroup shared with the kernel)
- Feature bucket peak is about 8.3 GB (measured); prep peaks at about 8.7 GB. Never run two heavy stages at once.
- Keep the kernel light: no big DataFrames in the notebook.
