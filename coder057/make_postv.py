#!/usr/bin/env python3
"""Generate postproc variants (CPU, zero GPU quota) from saved test probs.

Each variant is an independent kernel: change VARIANTS, re-run this script, push.
  - weights:      which prob files to blend (tags normalised: fold1 -> "1", baseline stays)
  - target_scale: multiply the LB-best per-class mask budget (1.0 = 3956 masks)
  - kernel_sources: which infer kernels must be attached for those prob files
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

BASE_TARGET = {"defect_a": 675, "defect_b": 570, "defect_c": 1440, "defect_d": 1271}
BASE_SOURCES = ["coder057/dl-infer-v2", "coder057/dl-infer-v3"]

VARIANTS = [
    # 3 strong models (fold4 comes from dl-infer-v4 once training finishes)
    dict(name="post-v7", weights={"baseline": 1.0, "1": 1.0, "4": 1.0}, scale=1.0,
         sources=BASE_SOURCES + ["coder057/dl-infer-v4"]),
    # CC-size sweep on the 4-model blend (blob-size floor untested on LB so far)
    dict(name="post-v13", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.0, mc_scale=0.55,
         sources=BASE_SOURCES + ["coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    dict(name="post-v14", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.0, mc_scale=1.6,
         sources=BASE_SOURCES + ["coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    # fold1ft (val 0.7296) + fold4ft fine-tuned + fold5: all upgraded strong members
    dict(name="post-v17", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.0,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-ft-fold4", "coder057/dl-infer-v5"]),
    # fold4/fold5 both hit val 0.7048/0.7049 -> up-weight them over baseline+fold1 in the blend
    dict(name="post-v15", weights={"baseline": 1.0, "1": 1.0, "4": 1.3, "5": 1.3}, scale=1.0,
         sources=BASE_SOURCES + ["coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    # fold1 fine-tuned +12ep (dl-train-v9 also emits test_probs_fold1.npy)
    dict(name="post-v16", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.0,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    # all four strong models (fold4 val 0.7048, fold5 pending): wider strong ensemble
    dict(name="post-v12", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.0,
         sources=BASE_SOURCES + ["coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    # clean A/B against post-v16: same members minus fold5 -> isolates fold5's contribution
    dict(name="post-v20", weights={"baseline": 1.0, "1": 1.0, "4": 1.0}, scale=1.0,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4"]),
    # fold4ft swapped in for fold4 (file test_probs_fold4.npy -> tag "4"), ft1 kept from train-v9
    dict(name="post-v21", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.0,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-probs-fold4ft", "coder057/dl-infer-v5"]),
    # higher mask budget on the LB-best blend (v16 won at 3744 masks; is more still better?)
    dict(name="post-v22", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.06,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    # all five folds + baseline, no ft (does fold-coverage beat the ft models?)
    dict(name="post-v23", weights={"baseline": 1.0, "1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0}, scale=1.0,
         sources=BASE_SOURCES + ["coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    # blob-size floor x1.6 on the LB-best blend (v14 hinted tighter blobs help)
    dict(name="post-v24", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.0, mc_scale=1.6,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5"]),
    # 5 models: fold4 AND ft4 both present (ft4 comes from dl-rename-ft4, tag ft4)
    dict(name="post-v25", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.0,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # same 5 models, budget x1.06 (v22 hinted +0.0003 at higher budget)
    dict(name="post-v26", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.06,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # up-weight the fine-tuned member (ft1 was the single biggest gain)
    dict(name="post-v27", weights={"baseline": 1.0, "1": 1.5, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.0,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # lower budget on the LB-best 4-model blend (low side never tested with the ft models)
    dict(name="post-v28", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=0.94,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-probs-fold4ft", "coder057/dl-infer-v5"]),
    # up-weight ft4 instead of ft1 (which ft model carries the gain?)
    dict(name="post-v29", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.5}, scale=1.0,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # v26 (0.64999) says the 5-model blend wants MORE masks than 3956 -> walk the budget up
    dict(name="post-v30", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.09,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # v26 (0.64999) says the 5-model blend wants MORE masks than 3956 -> walk the budget up
    dict(name="post-v31", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.12,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # v26 (0.64999) says the 5-model blend wants MORE masks than 3956 -> walk the budget up
    dict(name="post-v32", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.15,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # v26 (0.64999) says the 5-model blend wants MORE masks than 3956 -> walk the budget up
    dict(name="post-v33", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.18,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # v26 (0.64999) says the 5-model blend wants MORE masks than 3956 -> walk the budget up
    dict(name="post-v34", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.22,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # v26 (0.64999) says the 5-model blend wants MORE masks than 3956 -> walk the budget up
    dict(name="post-v35", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0, "ft4": 1.0}, scale=1.28,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-infer-v4", "coder057/dl-infer-v5", "coder057/dl-rename-ft4"]),
    # winning budget on v21's shape (ft4 instead of plain fold4)
    dict(name="post-v36", weights={"baseline": 1.0, "1": 1.0, "4": 1.0, "5": 1.0}, scale=1.12,
         sources=["coder057/dl-infer-v3", "coder057/dl-train-v9", "coder057/dl-probs-fold4ft", "coder057/dl-infer-v5"]),
    # budget sweep on the proven 2-model blend (is 3769 masks actually optimal?)
    dict(name="post-v8", weights={"baseline": 1.0, "1": 1.0}, scale=0.85, sources=BASE_SOURCES),
    dict(name="post-v9", weights={"baseline": 1.0, "1": 1.0}, scale=1.15, sources=BASE_SOURCES),
    # do the weak folds help at LOW weight, now that density is matched?
    dict(name="post-v10", weights={"baseline": 1.0, "1": 1.0, "2": 0.3, "3": 0.3}, scale=1.0, sources=BASE_SOURCES),
    # prob scales differ ~5x between models (thr 0.175 vs 0.908) -> standardise before averaging
    dict(name="post-v11", weights={"baseline": 1.0, "1": 1.0}, scale=1.0, mode="zscore", sources=BASE_SOURCES),
    # CROSS-ACCOUNT SHARE: bytescode0115's 4 folds (b1..b4) mounted as a dataset,
    # coder057 has no GPU left but still has 5+5 submission slots
    dict(name="post-v18", weights={"baseline": 1.0, "b1": 1.0, "b2": 1.0, "b3": 1.0, "b4": 1.0}, scale=1.0,
         sources=["coder057/dl-infer-v3"], datasets=["coder057/bytes-probs4"]),
    # + coder057's long-trained fold1 (30ep+12ep ft) = 6 strong members
    dict(name="post-v19", weights={"baseline": 1.0, "1": 1.0, "b1": 1.0, "b2": 1.0, "b3": 1.0, "b4": 1.0},
         scale=1.0, sources=["coder057/dl-infer-v3", "coder057/dl-train-v9"], datasets=["coder057/bytes-probs4"]),
]

NB_HEAD = r'''
import time, json
import numpy as np, pandas as pd, cv2
from pathlib import Path

t_start = time.time()
SETTINGS = __SETTINGS__
WEIGHTS = {k: float(v) for k, v in SETTINGS["weights"].items() if float(v) > 0}
MODE = SETTINGS.get("mode", "mean")  # "mean" = raw prob average, "zscore" = per-model/class standardised
TARGET = SETTINGS["target_masks"]
MC, MT = SETTINGS["min_component"], SETTINGS["min_pixels"]
BATCH = 256
DEFECT_TYPES = ["defect_a", "defect_b", "defect_c", "defect_d"]
ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
if not (ROOT / "sample_submission.csv").exists():  # Kaggle mounts the competition data at varying depths
    _hits = sorted(Path("/kaggle/input").rglob("sample_submission.csv"))
    if _hits:
        ROOT = _hits[0].parent
print("ROOT:", ROOT, flush=True)
assert (ROOT / "sample_submission.csv").exists(), f"competition data not mounted: {ROOT}"
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
    base = (probs_c >= thr).astype(np.uint8)
    out = np.zeros_like(base)
    for b in range(base.shape[0]):
        m = cc_filter_mask(base[b], min_component) if (min_component > 0 and base[b].any()) else base[b]
        if m.sum() >= min_pixels:
            out[b] = m
    return out


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
Q8 = {k: probs[k].dtype == np.uint8 for k in WEIGHTS}  # uint8 prob dumps are dequantised on the fly
n_img, C, H, W = next(iter(probs.values())).shape
wsum = float(sum(WEIGHTS.values()))
print(f"probs {n_img} x {C} x {H}x{W} | weights {WEIGHTS} | {time.time()-t_start:.0f}s", flush=True)


STATS = None
if MODE == "zscore":
    n_px = 0
    s1 = {k: np.zeros(C, np.float64) for k in WEIGHTS}
    s2 = {k: np.zeros(C, np.float64) for k in WEIGHTS}
    for i0 in range(0, n_img, BATCH):
        for k in WEIGHTS:
            piece = np.asarray(probs[k][i0:i0 + BATCH], dtype=np.float32) / (255.0 if Q8[k] else 1.0)
            s1[k] += piece.sum(axis=(0, 2, 3))
            s2[k] += np.square(piece).sum(axis=(0, 2, 3))
        n_px += piece.shape[0] * H * W
    STATS = {k: (s1[k] / n_px, np.sqrt(np.maximum(s2[k] / n_px - (s1[k] / n_px) ** 2, 1e-8))) for k in WEIGHTS}
    print("model stats (mean/std per class):", {k: (np.round(m, 4).tolist(), np.round(d, 4).tolist()) for k, (m, d) in STATS.items()}, flush=True)


def blend(i0, i1):
    acc = None
    for k, w in WEIGHTS.items():
        piece = np.asarray(probs[k][i0:i1], dtype=np.float32) / (255.0 if Q8[k] else 1.0)
        if MODE == "zscore":
            mu, sd = STATS[k]
            piece = (piece - mu.reshape(1, C, 1, 1)) / sd.reshape(1, C, 1, 1)
        acc = piece * w if acc is None else acc + piece * w
    return acc / wsum


mx = np.zeros((C, n_img), dtype=np.float32)
for i0 in range(0, n_img, BATCH):
    mx[:, i0:i0 + BATCH] = blend(i0, i0 + BATCH).max(axis=(2, 3)).T
print(f"per-image max probs done | {time.time()-t_start:.0f}s", flush=True)


def count_class(c, thr):
    n = 0
    for i0 in range(0, n_img, BATCH):
        n += int((class_masks(blend(i0, i0 + BATCH)[:, c], thr, MC[c], MT[c]).sum(axis=(1, 2)) > 0).sum())
    return n


thresholds = []
for c, dt in enumerate(DEFECT_TYPES):
    target = int(TARGET[dt])
    order = np.sort(mx[c])[::-1]
    thr = float(order[min(max(target - 1, 0), len(order) - 1)]) - 1e-4
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

rle_dict = {}
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

chk = pd.read_csv("/kaggle/working/submission.csv")
assert list(chk.columns) == list(sub_raw.columns)
assert len(chk) == len(sub_raw)
assert set(chk["image_id"]) == set(sub_raw["image_id"])
assert chk["mask_rle"].notna().all()
assert chk["mask_rle"].str.match(r"^(\s*|\d+( \d+)*)$").all()
print(f"VALIDATION PASSED | {time.time()-t_start:.0f}s", flush=True)
'''

for v in VARIANTS:
    out = HERE / v["name"]
    out.mkdir(exist_ok=True)
    settings = {
        "weights": v["weights"],
        "mode": v.get("mode", "mean"),
        "target_masks": {k: int(round(n * v["scale"])) for k, n in BASE_TARGET.items()},
        "min_component": [int(round(x * v.get("mc_scale", 1.0))) for x in (48, 32, 160, 256)],
        "min_pixels": [int(round(x * v.get("mc_scale", 1.0))) for x in (64, 0, 192, 384)],
    }
    cell_src = NB_HEAD.replace("__SETTINGS__", json.dumps(settings))
    cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell_src}]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{v['name']}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"coder057/dl-{v['name']}", "title": f"DL-{v['name']}", "code_file": f"{v['name']}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": v.get("datasets", []), "kernel_sources": v["sources"],
            "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{v['name']}: targets {list(settings['target_masks'].values())} total {sum(settings['target_masks'].values())}")
