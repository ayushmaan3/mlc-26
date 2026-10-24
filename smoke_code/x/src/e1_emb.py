"""Parallel path E: multilingual-e5-small embeddings for (1) kNN blocking and (2) a cosine feature.
  encode  : "query: <name> | <address>" for every record -> {split}_emb.f16 (memmap [N, 384] float16, L2-normalised)
            row order = {split}_records.parquet row order (ids in {split}_emb_ids.npy)
  knn     : within each country, each S1 -> top-K S2/S3 by cosine (GPU, tiled matmul) -> knn_{split}_{j}.parquet
            [s1, id, cos, erk]   (j = S1 bucket, same bucketing as cand2)
  recall  : train only: recall of cand2, kNN, and their union vs ground truth; new pairs per S1
  cos     : cosine for every existing cand2 pair -> embcos_{split}_{j}.parquet [s1, id, emb_cos, emb_rk]
            (joined automatically by s4_train stage 3 when present)
Every step skips outputs that already exist. Nothing big is held in RAM (memmap + chunks)."""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from common import WORK, done, timer
from s2_block import BUCKETS, cand_path

MODEL = os.environ.get("BER_EMB_MODEL", "/content/drive/MyDrive/MLC'26/models/multilingual-e5-small")
K = int(os.environ.get("BER_KNN_K", 20))
DIM, MAXLEN, BS = 384, 64, 1024
QB = 1024   # S1 rows per GPU block: scores tile = QB x 1M fp16 = 2 GB
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def emb_path(split):
    return WORK / f"{split}_emb.f16"


def ids_path(split):
    return WORK / f"{split}_emb_ids.npy"


def knn_path(split, j):
    return WORK / f"knn_{split}_{j}.parquet"


def embcos_path(split, j):
    return WORK / f"embcos_{split}_{j}.parquet"


def records(split):
    return pl.read_parquet(WORK / f"{split}_records.parquet", columns=["id", "src", "name", "addr", "country"])


def open_emb(split, mode="r"):
    n = len(np.load(ids_path(split)))
    return np.memmap(emb_path(split), dtype=np.float16, mode=mode, shape=(n, DIM))


NCHUNK = 8   # encode in 8 chunks; each finished chunk is saved (and copied to BER_EMB_SYNC if set)


def chunk_path(split, c):
    return WORK / f"{split}_emb_c{c}.f16"


@torch.no_grad()
def encode(split):
    """Resumable: records are encoded shortest-first in NCHUNK chunks. A finished chunk is written as
    {split}_emb_c{c}.f16 and, if BER_EMB_SYNC is set, copied there (Drive) at once, so a crash or a dead session
    loses at most one chunk. Chunks already present are skipped. Finally the chunks are assembled into
    {split}_emb.f16 in records row order (same layout as before)."""
    import shutil
    if done(emb_path(split)) and done(ids_path(split)):
        return
    sync = os.environ.get("BER_EMB_SYNC")
    with timer(f"encode {split}"):
        r = records(split)
        np.save(ids_path(split), r["id"].to_numpy())
        txt = ("query: " + r["name"] + " | " + r["addr"]).str.slice(0, 250).to_list()
        del r
        n = len(txt)
        order = np.argsort(np.fromiter((len(t) for t in txt), dtype=np.int32, count=n), kind="stable")
        bounds = np.linspace(0, n, NCHUNK + 1).astype(np.int64)
        tok = model = None
        for c in range(NCHUNK):
            if done(chunk_path(split, c)):
                print(f"  chunk {c} present, skipped", flush=True)
                continue
            if model is None:
                tok = AutoTokenizer.from_pretrained(MODEL)
                model = AutoModel.from_pretrained(MODEL).to(DEV).half().eval()
            idx_c = order[bounds[c]:bounds[c + 1]]
            tmp = chunk_path(split, c).with_suffix(".part")
            out = np.memmap(tmp, dtype=np.float16, mode="w+", shape=(len(idx_c), DIM))
            t0 = time.time()
            for k in range(0, len(idx_c), BS):
                idx = idx_c[k:k + BS]
                enc = tok([txt[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN,
                          return_tensors="pt").to(DEV)
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=1)
                out[k:k + len(idx)] = e.half().cpu().numpy()
                if k % (BS * 300) == 0:
                    el = time.time() - t0
                    print(f"  chunk {c + 1}/{NCHUNK}: {k:,}/{len(idx_c):,} {(k + BS) / max(el, 1e-6):.0f}/s "
                          f"chunk eta {el / (k + BS) * (len(idx_c) - k) / 60:.0f} min", flush=True)
            out.flush()
            del out
            os.replace(tmp, chunk_path(split, c))
            if sync:
                shutil.copy(chunk_path(split, c), os.path.join(sync, chunk_path(split, c).name + ".part"))
                os.replace(os.path.join(sync, chunk_path(split, c).name + ".part"), os.path.join(sync, chunk_path(split, c).name))
            print(f"  CHUNK {c + 1}/{NCHUNK} DONE ({time.time() - t0:.0f}s){' + saved to Drive' if sync else ''}", flush=True)
        # assemble in records row order (sorted writes -> near-sequential I/O)
        tmp = WORK / f"{split}_emb.f16.part"
        E = np.memmap(tmp, dtype=np.float16, mode="w+", shape=(n, DIM))
        for c in range(NCHUNK):
            pos = order[bounds[c]:bounds[c + 1]]
            C = np.memmap(chunk_path(split, c), dtype=np.float16, mode="r", shape=(len(pos), DIM))
            srt = np.argsort(pos)
            for i in range(0, len(pos), 1_000_000):
                sl = srt[i:i + 1_000_000]
                E[pos[sl]] = C[sl]
            del C
        E.flush()
        del E
        os.replace(tmp, emb_path(split))   # only a finished file gets the final name


@torch.no_grad()
def knn(split):
    if all(done(knn_path(split, j)) for j in range(BUCKETS)):
        return
    E = open_emb(split)
    r = pl.read_parquet(WORK / f"{split}_records.parquet", columns=["id", "src", "country"]).with_row_index("row")
    res = []
    for c in r["country"].unique().to_list():
        with timer(f"knn {split} {c}"):
            q = r.filter((pl.col("country") == c) & (pl.col("src") == 1))
            d = r.filter((pl.col("country") == c) & (pl.col("src") != 1))
            if q.height == 0 or d.height == 0:
                continue
            drow, did = d["row"].to_numpy(), d["id"].to_numpy()
            # database on GPU in chunks of 1M rows (fp16: ~0.77 GB per chunk)
            D = [torch.from_numpy(np.ascontiguousarray(E[np.sort(drow[i:i + 1_000_000])])).to(DEV)
                 for i in range(0, len(drow), 1_000_000)]
            dorder = np.concatenate([np.sort(drow[i:i + 1_000_000]) for i in range(0, len(drow), 1_000_000)])
            row2id = dict(zip(drow.tolist(), did.tolist()))
            did_sorted = np.fromiter((row2id[x] for x in dorder), dtype=np.int64, count=len(dorder))
            qrow, qid = q["row"].to_numpy(), q["id"].to_numpy()
            for i in range(0, len(qrow), QB):
                Q = torch.from_numpy(np.ascontiguousarray(E[qrow[i:i + QB]])).to(DEV)
                best_s, best_i, off = [], [], 0
                for Dc in D:
                    s = Q @ Dc.T
                    v, ix = s.topk(min(K, Dc.shape[0]), dim=1)
                    best_s.append(v.float()); best_i.append(ix + off); off += Dc.shape[0]
                v, ix = torch.cat(best_s, 1).topk(K, dim=1)
                gi = torch.gather(torch.cat(best_i, 1), 1, ix).cpu().numpy()
                v = v.cpu().numpy()
                res.append(pl.DataFrame({"s1": np.repeat(qid[i:i + QB], K), "id": did_sorted[gi.ravel()],
                                         "cos": v.ravel().astype(np.float32),
                                         "erk": np.tile(np.arange(1, K + 1, dtype=np.int16), len(gi))}))
            del D
            torch.cuda.empty_cache()
    res = pl.concat(res)
    for j in range(BUCKETS):
        res.filter(pl.col("s1") % BUCKETS == j).write_parquet(knn_path(split, j))


def recall():
    """Bucket by bucket (loading all 110M candidate pairs at once crashed a 12.7 GB runtime)."""
    tot = hc = n1 = ncand = 0
    hk, hu, nu = {5: 0, 10: 0, 20: 0}, {5: 0, 10: 0, 20: 0}, {5: 0, 10: 0, 20: 0}
    for j in range(BUCKETS):
        gt = pl.read_parquet(WORK / "train_gt.parquet").filter(pl.col("s1") % BUCKETS == j)
        cand = pl.read_parquet(cand_path("train", j), columns=["s1", "id"])
        kn = pl.read_parquet(knn_path("train", j))
        n1 += cand["s1"].n_unique(); ncand += cand.height; tot += gt.height
        hc += gt.join(cand, on=["s1", "id"], how="semi").height
        for k in (5, 10, 20):
            kk = kn.filter(pl.col("erk") <= k).select("s1", "id")
            u = pl.concat([cand, kk]).unique()
            hk[k] += gt.join(kk, on=["s1", "id"], how="semi").height
            hu[k] += gt.join(u, on=["s1", "id"], how="semi").height
            nu[k] += u.height - cand.height
        del cand, kn
    print(f"cand2 recall {hc / tot:.4f}")
    for k in (5, 10, 20):
        print(f"kNN@{k}: alone {hk[k] / tot:.4f} | union {hu[k] / tot:.4f} | new pairs/S1 {nu[k] / n1:.1f}", flush=True)


def cosine(split):
    E = open_emb(split)
    row = pl.DataFrame({"id": np.load(ids_path(split))}).with_row_index("row")
    for j in range(BUCKETS):
        if done(embcos_path(split, j)):
            continue
        with timer(f"cos {split} {j}"):
            c = (pl.read_parquet(cand_path(split, j), columns=["s1", "id"])
                 .join(row.rename({"id": "s1", "row": "r1"}), on="s1").join(row.rename({"row": "r2"}), on="id"))
            out = np.empty(c.height, dtype=np.float32)
            r1, r2 = c["r1"].to_numpy(), c["r2"].to_numpy()
            for i in range(0, c.height, 1_000_000):
                a = torch.from_numpy(E[r1[i:i + 1_000_000]]).to(DEV).float()
                b = torch.from_numpy(E[r2[i:i + 1_000_000]]).to(DEV).float()
                out[i:i + 1_000_000] = (a * b).sum(1).cpu().numpy()
            (c.select("s1", "id").with_columns(pl.Series("emb_cos", out))
             .with_columns(pl.col("emb_cos").rank("min", descending=True).over("s1").cast(pl.Int16).alias("emb_rk"))
             .write_parquet(embcos_path(split, j)))


if __name__ == "__main__":
    # usage: e1_emb.py encode|knn|recall|cos <split...>
    step, splits = sys.argv[1], sys.argv[2:] or ["train", "test"]
    for s in splits:
        {"encode": encode, "knn": knn, "cos": cosine}.get(step, lambda _: None)(s)
    if step == "recall":
        recall()
