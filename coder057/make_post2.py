#!/usr/bin/env python3
"""Generate post-v2: CPU re-postprocess combining saved test probs from dl-infer-v2 (folds 1-3)
and dl-infer-v3 (baseline model). Change SETTINGS + push a new version for a free variant.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "post-v2"
OUT.mkdir(exist_ok=True)

SETTINGS = {
    "weights": {"baseline": 1.0, "1": 1.0, "2": 0.0, "3": 0.0},
    "thresholds": [0.175, 0.05, 0.375, 0.2],
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
THR, MC, MT = SETTINGS["thresholds"], SETTINGS["min_component"], SETTINGS["min_pixels"]
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


def postprocess(probs, thresholds, min_component, min_pixels):
    B, C, H, W = probs.shape
    out = np.zeros((B, C, H, W), dtype=np.uint8)
    for c in range(C):
        base = (probs[:, c].astype(np.float32) >= thresholds[c]).astype(np.uint8)
        if min_component[c] > 0 and base.any():
            for b in range(B):
                base[b] = cc_filter_mask(base[b], min_component[c])
        for b in range(B):
            if base[b].sum() >= min_pixels[c]:
                out[b, c] = base[b]
    return out
''')

code(r'''
sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()

found = {}
for p in Path("/kaggle/input").rglob("test_probs_*.npy"):
    tag = p.name[len("test_probs_"):-len(".npy")]
    tag = tag[len("fold"):] if tag.startswith("fold") else tag  # fold1 -> 1, baseline stays
    found[tag] = p
print("prob files:", {k: str(v.parent) for k, v in sorted(found.items())}, flush=True)
missing = [k for k in WEIGHTS if k not in found]
assert not missing, f"missing prob files for {missing}; found {sorted(found)}"
probs = {k: np.load(found[k], mmap_mode="r") for k in WEIGHTS}
n_img, C, H, W = next(iter(probs.values())).shape
wsum = float(sum(WEIGHTS.values()))
print(f"probs {n_img} imgs x {C} x {H}x{W} | weights {WEIGHTS} | {time.time()-t_start:.0f}s", flush=True)

rle_dict = {}
n_masks = 0
for i0 in range(0, n_img, BATCH):
    acc = None
    for k, w in WEIGHTS.items():
        piece = np.asarray(probs[k][i0:i0 + BATCH], dtype=np.float32)
        acc = piece * w if acc is None else acc + piece * w
    pred = postprocess(acc / wsum, THR, MC, MT)
    n_masks += int((pred.sum(axis=(2, 3)) > 0).sum())
    for b in range(pred.shape[0]):
        img_id = str(test_ids[i0 + b])
        for c, dt in enumerate(DEFECT_TYPES):
            m = pred[b, c]
            rle_dict[(img_id, dt)] = rle_encode(m) if m.sum() > 0 else " "
    if (i0 // BATCH) % 4 == 0:
        print(f"  postproc {min(i0+BATCH, n_img)}/{n_img} | {time.time()-t_start:.0f}s", flush=True)

sub = sub_raw.copy()
sub["mask_rle"] = [rle_dict.get((str(r.image_id), str(r.defect_type)), " ") for r in sub.itertuples(index=False)]
sub = sub[list(sub_raw.columns)]
sub.to_csv("/kaggle/working/submission.csv", index=False)
non_empty = int((sub["mask_rle"] != " ").sum())
print(f"\nsubmission.csv | rows {len(sub)} | masks {non_empty} ({non_empty/len(sub):.2%})", flush=True)
print("per class masks:", sub[sub.mask_rle != " "].defect_type.value_counts().reindex(DEFECT_TYPES).to_dict(), flush=True)

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
(OUT / "post-v2.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "coder057/dl-post-v2", "title": "DL-post-v2", "code_file": "post-v2.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [], "kernel_sources": ["coder057/dl-infer-v2", "coder057/dl-infer-v3"],
        "competition_sources": ["industrial-surface-defect-detection"],
        "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("written:", OUT / "post-v2.ipynb")
