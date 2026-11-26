#!/usr/bin/env python3
"""Generate post-v4: per-class threshold calibration to a TARGET MASK BUDGET.

Why: LB history (3956 masks -> 0.62361, 4930 -> 0.59838, 6537 -> 0.55949) says the
best submission's density is the one that scored. fold1 probs at the same fixed
thresholds produce 5479 masks, i.e. much denser. So search per-class thresholds that
reproduce the LB-best per-class mask counts, then postprocess.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "post-v4"
OUT.mkdir(exist_ok=True)

SETTINGS = {
    "weights": {"1": 1.0},
    # per-class mask counts of the LB-best submission (sweep, 0.62361)
    "target_masks": {"defect_a": 675, "defect_b": 570, "defect_c": 1440, "defect_d": 1271},
    "min_component": [48, 32, 160, 256],
    "min_pixels": [64, 0, 192, 384],
}

cells = []


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s})


code(r'''
import time, json
import numpy as np, pandas as pd, cv2
from pathlib import Path

t_start = time.time()
SETTINGS = ''' + json.dumps(SETTINGS) + r'''
WEIGHTS = {k: float(v) for k, v in SETTINGS["weights"].items() if float(v) > 0}
TARGET = SETTINGS["target_masks"]
MC, MT = SETTINGS["min_component"], SETTINGS["min_pixels"]
BATCH = 256
DEFECT_TYPES = ["defect_a", "defect_b", "defect_c", "defect_d"]
ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
SAMPLE = ROOT / "sample_submission.csv"
print("SETTINGS:", json.dumps(SETTINGS), flush=True)


def rle_encode(mask):
    pixels = (np.asarray(mask) > 0).flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))


def cc_filter_mask(mask, min_component):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component:
            out[labels == i] = 1
    return out


def class_masks(probs_c, thr, min_component, min_pixels):
    """(B,H,W) float32 probs for ONE class -> (B,H,W) uint8 kept masks."""
    base = (probs_c >= thr).astype(np.uint8)
    out = np.zeros_like(base)
    for b in range(base.shape[0]):
        m = cc_filter_mask(base[b], min_component) if (min_component > 0 and base[b].any()) else base[b]
        if m.sum() >= min_pixels:
            out[b] = m
    return out
''')


code(r'''
sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()

found = {}
for p in Path("/kaggle/input").rglob("test_probs_*.npy"):
    tag = p.name[len("test_probs_"):-len(".npy")]
    tag = tag[len("fold"):] if tag.startswith("fold") else tag  # fold1 -> 1
    found[tag] = p
print("prob files:", {k: str(v.parent) for k, v in sorted(found.items())}, flush=True)
missing = [k for k in WEIGHTS if k not in found]
assert not missing, f"missing {missing}; found {sorted(found)}"
probs = {k: np.load(found[k], mmap_mode="r") for k in WEIGHTS}
n_img, C, H, W = next(iter(probs.values())).shape
wsum = float(sum(WEIGHTS.values()))
print(f"probs {n_img} x {C} x {H}x{W} | weights {WEIGHTS} | {time.time()-t_start:.0f}s", flush=True)


def blend(i0, i1):
    acc = None
    for k, w in WEIGHTS.items():
        piece = np.asarray(probs[k][i0:i1], dtype=np.float32)
        acc = piece * w if acc is None else acc + piece * w
    return acc / wsum


# per-image max prob per class (for cheap threshold targeting)
mx = np.zeros((C, n_img), dtype=np.float32)
for i0 in range(0, n_img, BATCH):
    bl = blend(i0, i0 + BATCH)
    mx[:, i0:i0 + BATCH] = bl.max(axis=(2, 3)).T
print(f"per-image max probs done | {time.time()-t_start:.0f}s", flush=True)


def count_class(c, thr):
    n = 0
    for i0 in range(0, n_img, BATCH):
        m = class_masks(blend(i0, i0 + BATCH)[:, c], thr, MC[c], MT[c])
        n += int((m.sum(axis=(1, 2)) > 0).sum())
    return n


# initial thr = k-th largest per-image max (guarantees raw count >= target)
thresholds = []
for c, dt in enumerate(DEFECT_TYPES):
    target = int(TARGET[dt])
    order = np.sort(mx[c])[::-1]
    thr = float(order[min(max(target - 1, 0), len(order) - 1)]) - 1e-4
    # CC + min_pixels can only drop masks -> lower thr until count reaches target (max 4 tries)
    for attempt in range(4):
        cnt = count_class(c, thr)
        print(f"  {dt} try{attempt}: thr={thr:.4f} masks={cnt} target={target} | {time.time()-t_start:.0f}s", flush=True)
        if cnt >= target or attempt == 3:
            break
        need = target - cnt
        idx = min(target + need - 1, len(order) - 1)
        thr = float(order[max(idx - 1, 0)]) - 1e-4
    thresholds.append(round(thr, 4))
print(f"CALIBRATED thresholds {thresholds} | {time.time()-t_start:.0f}s", flush=True)
''')


code(r'''
rle_dict = {}
n_masks = 0
for i0 in range(0, n_img, BATCH):
    bl = blend(i0, i0 + BATCH)
    for c, dt in enumerate(DEFECT_TYPES):
        m = class_masks(bl[:, c], thresholds[c], MC[c], MT[c])
        for b in range(m.shape[0]):
            if m[b].sum() > 0:
                rle_dict[(str(test_ids[i0 + b]), dt)] = rle_encode(m[b])
    if (i0 // BATCH) % 4 == 0:
        print(f"  postproc {min(i0+BATCH, n_img)}/{n_img} | {time.time()-t_start:.0f}s", flush=True)

sub = sub_raw.copy()
sub["mask_rle"] = [rle_dict.get((str(r.image_id), str(r.defect_type)), " ") for r in sub.itertuples(index=False)]
sub = sub[list(sub_raw.columns)]
sub.to_csv("/kaggle/working/submission.csv", index=False)
non_empty = int((sub["mask_rle"] != " ").sum())
print(f"\nsubmission.csv | rows {len(sub)} | masks {non_empty} ({non_empty/len(sub):.2%})", flush=True)
print("per class masks:", sub[sub.mask_rle != " "].defect_type.value_counts().reindex(DEFECT_TYPES).to_dict(), flush=True)
print("thresholds:", thresholds, flush=True)

chk = pd.read_csv("/kaggle/working/submission.csv")
assert list(chk.columns) == list(sub_raw.columns)
assert len(chk) == len(sub_raw)
assert set(chk["image_id"]) == set(sub_raw["image_id"])
assert chk["mask_rle"].notna().all()
assert chk["mask_rle"].str.match(r"^(\s*|\d+( \d+)*)$").all()
print(f"VALIDATION PASSED | {time.time()-t_start:.0f}s", flush=True)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "post-v4.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "coder057/dl-post-v4", "title": "DL-post-v4", "code_file": "post-v4.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [], "kernel_sources": ["coder057/dl-infer-v2"],
        "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("written:", OUT / "post-v4.ipynb")
