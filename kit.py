"""Hackathon kit. `from kit import *` at the top of every notebook.

ROOT          persistent dir (Colab: Drive, SageMaker/local: ~/amzml, override with AMZ_ROOT)
cached()      compute once, load from disk forever after  -> cheap re-runs
shard_map()   resumable batched work (inference, OCR, embeddings, downloads) -> crash loses <1 shard
Ckpt          time-based checkpoints for custom torch loops (HF Trainer: see PLAYBOOK.md)
check_sub()   validate submission before burning 1 of 15 uploads
mem(), seed_all()
"""
import gc, glob, os, pickle, random, time
from pathlib import Path


def _root():
    if os.path.exists("/content"):  # Colab
        if not os.path.exists("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        r = "/content/drive/MyDrive/amzml"
    else:
        r = os.environ.get("AMZ_ROOT", os.path.expanduser("~/amzml"))
    Path(r).mkdir(parents=True, exist_ok=True)
    return Path(r)


ROOT = _root()


def _dump(obj, p):
    # write to tmp then rename: a crash mid-write never leaves a corrupt file behind
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=5)
    os.replace(tmp, p)


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def cached(name, fn):
    """Put every config knob that changes the output IN the name, e.g. f"emb_{model}_{maxlen}"."""
    p = ROOT / "cache" / f"{name}.pkl"
    if p.exists():
        return _load(p)
    out = fn()
    _dump(out, p)
    return out


def shard_map(items, fn, name, size=2000):
    """fn(list) -> list of same length. Done shards are skipped on re-run.
    items order must be stable for a given name; change name if items change."""
    d = ROOT / "shards" / name
    paths = []
    for i in range(0, len(items), size):
        p = d / f"{i:09d}.pkl"
        if not p.exists():
            t = time.time()
            out = fn(items[i:i + size])
            assert len(out) == len(items[i:i + size]), "fn must return one output per item"
            _dump(out, p)
            print(f"[{name}] {i + len(out)}/{len(items)}  {time.time() - t:.0f}s", flush=True)
        paths.append(p)
    return [x for p in paths for x in _load(p)]


class Ckpt:
    """ck = Ckpt("run07"); start = ck.load(model, opt, sched)
    in loop:  ck.maybe_save(step, model, opt, sched)      # every `every_min` minutes
    at end:   ck.maybe_save(step, model, opt, sched, force=True)
    Saves only trainable params + buffers (LoRA -> MBs, not GBs). Keeps last `keep`."""

    def __init__(self, name, every_min=20, keep=2):
        self.d = ROOT / "ckpt" / name
        self.every, self.keep, self.last = every_min * 60, keep, time.time()

    def load(self, model, opt=None, sched=None):
        import torch
        files = sorted(glob.glob(str(self.d / "*.pt")))
        if not files:
            return 0
        s = torch.load(files[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"], strict=False)
        if opt and s.get("opt"):
            opt.load_state_dict(s["opt"])
        if sched and s.get("sched"):
            sched.load_state_dict(s["sched"])
        random.setstate(s["py_rng"]); torch.set_rng_state(s["torch_rng"])
        print(f"resumed {files[-1]} at step {s['step']}")
        return s["step"] + 1

    def maybe_save(self, step, model, opt=None, sched=None, force=False):
        if not force and time.time() - self.last < self.every:
            return
        import torch
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        keep |= {n for n, _ in model.named_buffers()}
        state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
        self.d.mkdir(parents=True, exist_ok=True)
        p = self.d / f"{step:09d}.pt"
        torch.save({"step": step, "model": state,
                    "opt": opt.state_dict() if opt else None,
                    "sched": sched.state_dict() if sched else None,
                    "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()},
                   p.with_suffix(".tmp"))
        os.replace(p.with_suffix(".tmp"), p)
        for old in sorted(glob.glob(str(self.d / "*.pt")))[:-self.keep]:
            os.remove(old)
        self.last = time.time()
        print(f"ckpt {p}", flush=True)


def check_sub(sub, test, id_col, pred_cols):
    """Raises on anything that would get a submission rejected or silently scored wrong."""
    assert len(sub) == len(test), f"rows {len(sub)} != test {len(test)}"
    assert sub[id_col].is_unique, "duplicate ids"
    assert set(sub[id_col]) == set(test[id_col]), "id set differs from test"
    for c in pred_cols:
        assert c in sub.columns, f"missing column {c}"
        assert not sub[c].isna().any(), f"NaN in {c}"
    print("submission OK:", sub.shape)


def mem():
    import psutil
    vm = psutil.virtual_memory()
    s = f"RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.1f}G"
    try:
        import torch
        if torch.cuda.is_available():
            s += f" | GPU {torch.cuda.memory_allocated() / 2**30:.1f}G alloc, peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
    except ImportError:
        pass
    import shutil
    s += f" | disk free {shutil.disk_usage(ROOT).free / 2**30:.0f}G"
    print(s)
    gc.collect()


def seed_all(s=42):
    import numpy as np
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


if __name__ == "__main__":  # self-check: python kit.py   (set AMZ_ROOT to a temp dir)
    import pandas as pd
    calls = []
    assert cached("t", lambda: calls.append(1) or 5) == 5 and cached("t", lambda: 1 / 0) == 5
    f = lambda xs: calls.append(1) or [x * 2 for x in xs]
    n = len(calls)
    assert shard_map(list(range(10)), f, "t", size=3) == [x * 2 for x in range(10)]
    assert len(calls) - n == 4
    assert shard_map(list(range(10)), f, "t", size=3) == [x * 2 for x in range(10)]
    assert len(calls) - n == 4, "re-run must not recompute"
    t = pd.DataFrame({"id": [1, 2]}); check_sub(pd.DataFrame({"id": [2, 1], "y": [0, 1]}), t, "id", ["y"])
    try:
        check_sub(pd.DataFrame({"id": [1, 1], "y": [0, 1]}), t, "id", ["y"]); raise SystemExit("bad")
    except AssertionError:
        pass
    import torch
    m = torch.nn.Linear(2, 2); o = torch.optim.AdamW(m.parameters())
    ck = Ckpt("t"); ck.maybe_save(7, m, o, force=True)
    m2 = torch.nn.Linear(2, 2); assert Ckpt("t").load(m2, torch.optim.AdamW(m2.parameters())) == 8
    assert torch.equal(m.weight, m2.weight)
    mem(); print("kit OK")
