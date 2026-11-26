#!/usr/bin/env python3
"""Public export kernel: copy another account's kernel-output probs into a PUBLIC kernel output so a
different account can mount them via kernel_sources (private kernels are not mountable cross-account).

usage: python3 make_export.py <acct> <src_owner>/<src_slug> <slug> [public]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ACCT, SRC, SLUG = sys.argv[1], sys.argv[2], sys.argv[3]
PUBLIC = len(sys.argv) > 4 and sys.argv[4] == "public"

CODE = '''import shutil, time
from pathlib import Path

t0 = time.time()
srcs = sorted(Path("/kaggle/input").rglob("test_probs_*.npy"))
assert srcs, "no test_probs_*.npy in mounted sources"
for p in srcs:
    dst = Path("/kaggle/working") / p.name
    shutil.copy(p, dst)
    print(f"{p.name}: {dst.stat().st_size/1e6:.0f}MB | {time.time()-t0:.0f}s", flush=True)
print("export done", flush=True)
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
        "language": "python", "kernel_type": "notebook", "is_private": not PUBLIC,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [], "kernel_sources": [SRC],
        "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
(D / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
compile(CODE, "export", "exec")
print(f"written {D} | id {meta['id']} | public={PUBLIC} | src={SRC}")
