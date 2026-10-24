"""Preflight for run E (it may be launched by a teammate): opens every input the run reads (parquet footer, npy header),
checks embedding sizes, 2 GPUs, the original e5 (in the inputs, or Hugging Face reachable), disk and RAM.
Prints one line 'PREFLIGHT OK: ...' or 'PREFLIGHT FAILED' + the full list of problems (exit 1), in about a minute.
env: BER_WORK (linked inputs), E5 (original e5 dir found in the inputs, may be empty)"""
import os
import shutil
import urllib.request

import numpy as np
import polars as pl
import torch

W = os.environ["BER_WORK"]
need = ["train_gt.parquet", "test_source1.tsv", "validate_submission.py"]
for s in ("train", "test"):
    need += [f"{s}_{x}.parquet" for x in ("records", "norm2", "norm3", "idf_ah", "idf_ch")] + [f"{s}_emb.f16", f"{s}_emb_ids.npy"]
    need += [f"{s}_{b}_{j}.parquet" for b in ("cand2", "feat2") for j in range(8)]
    need += [f"{b}_{s}_{j}.parquet" for b in ("knn", "knnft", "ce_v3") for j in range(8)]
SRC = {"knnft": "Output of KAGGLE_runC (Add Input -> Your Work -> Notebooks)", "knn_": "dataset mlc26-emb",
       "_emb": "dataset mlc26-emb", "ce_v3": "dataset mlc26-v3"}

bad = []
for f in need:
    p = os.path.join(W, f)
    try:
        if f.endswith(".parquet"):
            pl.read_parquet_schema(p)
        elif f.endswith(".npy"):
            np.load(p, mmap_mode="r")
        else:
            with open(p, "rb") as fh:
                fh.read(16)
    except Exception as e:
        bad.append(f"  {f}: {type(e).__name__}  (comes from {next((v for k, v in SRC.items() if k in f), 'dataset mlc26-core')})")
for s in ("train", "test"):
    try:
        n, sz = len(np.load(f"{W}/{s}_emb_ids.npy")), os.path.getsize(f"{W}/{s}_emb.f16")
        if sz != n * 384 * 2:
            bad.append(f"  {s}_emb.f16 is {sz:,} bytes, expected {n * 384 * 2:,} (incomplete mlc26-emb?)")
    except Exception:
        pass   # already listed above
ng = torch.cuda.device_count()
MIN_GPUS = int(os.environ.get("BER_MIN_GPUS", 2))   # local rehearsal on a 1-GPU laptop: 1
if ng < MIN_GPUS:
    bad.append(f"  GPUs visible: {ng}, need {MIN_GPUS} (Session options -> Accelerator -> GPU T4 x2)")
e5 = os.environ.get("E5", "")
if e5 and os.path.exists(os.path.join(e5, "model.safetensors")):
    e5msg = f"original e5 in the inputs ({e5})"
else:
    try:
        urllib.request.urlopen("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/config.json", timeout=15)
        e5msg = "original e5 not in the inputs, Hugging Face reachable -> will download it"
    except Exception as e:
        bad.append(f"  original e5: not in the inputs and Hugging Face unreachable ({type(e).__name__}) -> Session options -> Internet On")
        e5msg = "MISSING"
free = shutil.disk_usage(W).free / 2**30
try:
    ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.0f} GB"
except (AttributeError, ValueError):   # not Linux (local rehearsal)
    ram = "n/a"
if bad:
    print(f"PREFLIGHT FAILED: {len(bad)} problem(s). Fix them and run again:", *bad, sep="\n", flush=True)
    raise SystemExit(1)
print(f"PREFLIGHT OK: all {len(need)} input files found and readable | GPUs {ng} ({torch.cuda.get_device_name(0)}) | {e5msg}"
      f" | disk free {free:.0f} GB | RAM {ram}", flush=True)
