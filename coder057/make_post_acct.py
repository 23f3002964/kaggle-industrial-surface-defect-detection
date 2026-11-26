#!/usr/bin/env python3
"""Generate postproc kernels for an arbitrary account that mounts <acct>/dl-infer-all.

usage: python3 make_post_acct.py <account>
Reuses the NB_HEAD notebook template from make_postv.py (single source of truth).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
acct = sys.argv[1]
src = (HERE / "make_postv.py").read_text()
NB_HEAD = src.split("NB_HEAD = r'''", 1)[1].split("\n'''", 1)[0]
BASE_TARGET = {"defect_a": 675, "defect_b": 570, "defect_c": 1440, "defect_d": 1271}
INFER = [f"{acct}/dl-infer-all"]
FOLDS = {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}  # fold5 skipped: quota cannot fit 5 x 5.2h
# NOTE: no baseline member here (cross-account model ref is not mountable); the baseline joins the
# blend on the coder057 side via dl-post-v18/v19, which read these same probs from a dataset.

VARIANTS = [
    # pure 4-fold ensemble at the LB-calibrated budget
    dict(name="acct-a", weights=FOLDS, scale=1.0),
    # budget nudged down (coder057 optimum sat at 3769/3956 masks)
    dict(name="acct-b", weights=FOLDS, scale=0.95),
    # budget nudged up
    dict(name="acct-c", weights=FOLDS, scale=1.04),
    # stricter blob floor
    dict(name="acct-d", weights=FOLDS, scale=1.0, mc_scale=1.6),
    # looser blob floor
    dict(name="acct-e", weights=FOLDS, scale=1.0, mc_scale=0.55),
]

for v in VARIANTS:
    slug = f"dl-post-{v['name']}"
    out = HERE / f"acct-{acct}-{slug}"
    out.mkdir(exist_ok=True)
    mc = v.get("mc_scale", 1.0)
    settings = {
        "weights": v["weights"],
        "mode": v.get("mode", "mean"),
        "target_masks": {k: int(round(n * v["scale"])) for k, n in BASE_TARGET.items()},
        "min_component": [int(round(x * mc)) for x in (48, 32, 160, 256)],
        "min_pixels": [int(round(x * mc)) for x in (64, 0, 192, 384)],
    }
    cell = NB_HEAD.replace("__SETTINGS__", json.dumps(settings))
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{acct}/{slug}", "title": f"DL-post-{v['name']}", "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": [], "kernel_sources": INFER,
            "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{v['name']}: weights {list(v['weights'])} | masks {sum(settings['target_masks'].values())} | mc {settings['min_component']}")
print("kernels written for", acct)
