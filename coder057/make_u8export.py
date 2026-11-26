#!/usr/bin/env python3
"""Quantise big float32 test_probs dumps to uint8 so they can be downloaded + re-uploaded as a public
dataset (cross-account kernel outputs are not mountable, datasets are).

usage: python3 make_u8export.py <acct> <src_slug> <slug>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ACCT, SRC, SLUG = sys.argv[1], sys.argv[2], sys.argv[3]

CODE = '''import time
import numpy as np
from pathlib import Path

t0 = time.time()
out = Path("/kaggle/working")
srcs = sorted(Path("/kaggle/input").rglob("test_probs_*.npy"))
assert srcs, "no test_probs_*.npy mounted"
for p in srcs:
    a = np.load(p, mmap_mode="r")
    dst = out / p.name
    q = np.lib.format.open_memmap(dst, mode="w+", dtype=np.uint8, shape=a.shape)
    for i in range(0, a.shape[0], 1024):
        q[i:i + 1024] = (np.clip(np.asarray(a[i:i + 1024]), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    q.flush()
    del q
    print(f"{p.name} uint8 {a.shape} {dst.stat().st_size/1e6:.0f}MB | {time.time()-t0:.0f}s", flush=True)
print("u8 export done", flush=True)
'''

nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": SLUG, "metadata": {}, "outputs": [],
                 "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
D = HERE / f"x-{ACCT}-{SLUG}"
D.mkdir(exist_ok=True)
(D / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"{ACCT}/{SLUG}", "title": SLUG.upper(), "code_file": f"{SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [], "kernel_sources": [f"{ACCT}/{SRC}"],
        "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
(D / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
compile(CODE, "u8", "exec")
print(f"written {D} | id {meta['id']} | src {ACCT}/{SRC}")
