#!/usr/bin/env python3
"""Generate infer-v1: 3-fold TTA ensemble inference -> submission.csv (no training).
Models come from datasets coder057/fold1-model + coder057/folds23-models.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "infer-v2"
OUT.mkdir(exist_ok=True)

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s})


md("# DL-infer-v2 — 3-fold TTA ensemble -> submission\nNo training. Loads FOLD1/2/3 models, TTA probs, threshold selection on each fold's own val split, chunked postprocess + RLE.")

code("!pip install -q iterative-stratification")

code(r'''
import os, gc, time, json
import numpy as np, pandas as pd, cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

t_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| GPUs:", torch.cuda.device_count(), flush=True)

ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
TRAIN_DIR, TEST_DIR = ROOT / "train", ROOT / "test"
CSV_PATH, SAMPLE_SUB_PATH = ROOT / "train.csv", ROOT / "sample_submission.csv"
DEFECT_TYPES = ["defect_a", "defect_b", "defect_c", "defect_d"]
CONFIG = {"seed": 42}
THRESH_GRID = np.round(np.arange(0.05, 0.801, 0.025), 3)
TUNED = {"thresholds": [0.175, 0.05, 0.375, 0.2],
         "min_component": [48, 32, 160, 256],
         "min_pixels": [64, 0, 192, 384]}
BATCH = 64
''')

code(r'''
# ===== model architecture (must match the pickled modules) =====
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.down(x)


class AttentionGate(nn.Module):
    def __init__(self, gating_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(gating_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.W_x = nn.Sequential(nn.Conv2d(skip_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 1))
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attention(x, skip)], dim=1))


class AttentionDefectUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(DoubleConv(in_channels, 32), DownBlock(32, 64),
                                     DownBlock(64, 128), DownBlock(128, 256))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.decoder = nn.ModuleList([AttentionUpBlock(512, 256, 256), AttentionUpBlock(256, 128, 128),
                                      AttentionUpBlock(128, 64, 64), AttentionUpBlock(64, 32, 32)])
        self.outc = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x); skips.append(x)
        x = self.bottleneck(x)
        for layer, skip in zip(self.decoder, reversed(skips)):
            x = layer(x, skip)
        return self.outc(x)
''')

code(r'''
# ===== RLE / postprocess / metric =====
def rle_decode(rle_string, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle_string) or not str(rle_string).strip():
        return mask.reshape((height, width), order="F")
    rle = np.asarray(str(rle_string).split(), dtype=np.int64)
    starts, lengths = rle[0::2] - 1, rle[1::2]
    for s, l in zip(starts, starts + lengths):
        mask[s:l] = 1
    return mask.reshape((height, width), order="F")


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


def competition_score(pred_u8, tgt_u8):
    N, C, H, W = pred_u8.shape
    p = pred_u8.reshape(N, C, H * W).astype(np.float64)
    t = (tgt_u8.reshape(N, C, H * W) > 0).astype(np.float64)
    pred_px, tgt_px = p.sum(-1), t.sum(-1)
    inter = (p * t).sum(-1)
    eps = 1e-7
    pos, emp = tgt_px > 0, tgt_px == 0
    dice = (2 * inter + eps) / (pred_px + tgt_px + eps)
    fg = dice[pos].mean() if pos.any() else 0.0
    fps = 1.0 - (pred_px[emp] / (H * W)).mean() if emp.any() else 1.0
    ea = (pred_px[emp] == 0).mean() if emp.any() else 1.0
    return 0.7 * fg + 0.2 * fps + 0.1 * ea


def best_thresholds(probs, tgt):
    """Threshold-only sweep per class (CC/min_pixels applied later at final postproc)."""
    N, C, H, W = probs.shape
    tgt_px = (tgt.reshape(N, C, H * W) > 0).sum(-1).astype(np.int64)
    best = []
    for c in range(C):
        p = probs[:, c].reshape(N, H * W)
        t = tgt[:, c].reshape(N, H * W) > 0
        tpx, pos, emp = tgt_px[:, c], tgt_px[:, c] > 0, tgt_px[:, c] == 0
        scores = []
        for thr in THRESH_GRID:
            pr = p >= np.float16(thr)
            ppx = pr.sum(1).astype(np.float64)
            inter = (pr & t).sum(1).astype(np.float64)
            dice = (2 * inter + 1e-7) / (ppx + tpx + 1e-7)
            fg = dice[pos].mean() if pos.any() else 0.0
            fps = 1.0 - (ppx[emp] / (H * W)).mean() if emp.any() else 1.0
            ea = (ppx[emp] == 0).mean() if emp.any() else 1.0
            scores.append(0.7 * fg + 0.2 * fps + 0.1 * ea)
        i = int(np.argmax(scores))
        best.append(float(THRESH_GRID[i]))
        print(f"    class {DEFECT_TYPES[c]}: thr={THRESH_GRID[i]:.3f} score={scores[i]:.4f}", flush=True)
    return best
''')

code(r'''
# ===== STAGE 1/5: load data + splits =====
df = pd.read_csv(CSV_PATH)
image_df = pd.DataFrame({"image_id": df["image_id"].drop_duplicates().values})
for d in DEFECT_TYPES:
    pos = df.loc[(df["defect_type"] == d) & df["mask_rle"].notna()
                 & (df["mask_rle"].astype(str).str.strip() != ""), "image_id"].unique()
    image_df[d] = image_df["image_id"].isin(pos).astype(int)
X = image_df["image_id"].values
y = image_df[DEFECT_TYPES].values
H, W = int(df["height"].iloc[0]), int(df["width"].iloc[0])
print(f"train images {len(X)} | H,W {H},{W} | pos rate {y.mean(0).round(3)}", flush=True)

sub_raw = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sub_raw["image_id"].unique()


def load_images(image_dir, ids, desc):
    arr = np.empty((len(ids), H, W), dtype=np.uint8)
    for i, iid in enumerate(tqdm(ids, desc=desc, dynamic_ncols=True)):
        name = iid if Path(iid).suffix else f"{iid}.jpg"
        with Image.open(image_dir / name) as im:
            a = np.asarray(im.convert("L"), dtype=np.uint8)
        if a.shape != (H, W):
            raise ValueError(f"image {iid} shape {a.shape} != {(H, W)}")
        arr[i] = a
    return arr


train_u8 = load_images(TRAIN_DIR, X, "train imgs")
test_u8 = load_images(TEST_DIR, test_ids, "test imgs")
print(f"train_u8 {train_u8.nbytes/1e9:.2f}GB | test_u8 {test_u8.nbytes/1e9:.2f}GB | {time.time()-t_start:.0f}s", flush=True)

kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
folds = list(kfold.split(X, y))[:3]
print("folds ready:", [len(v) for _, v in folds], flush=True)
''')

code(r'''
# ===== STAGE 2/5: models + val probs (TTA) + threshold sweep =====
model_paths = {int(p.stem[-1]): p for p in sorted(Path("/kaggle/input").rglob("ATTENTION-UNET-FOLD*.pth"))}
print("models found:", {k: str(v) for k, v in sorted(model_paths.items())}, flush=True)
assert len(model_paths) >= 3, f"need 3 models, found {sorted(model_paths)}"


@torch.inference_mode()
def probs_tta(model, u8, tag=""):
    out = np.empty((len(u8), 4, H, W), dtype=np.float16)
    t0 = time.time()
    for s in range(0, len(u8), BATCH):
        x = torch.from_numpy(u8[s:s + BATCH]).unsqueeze(1).float().div_(255).to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                logits = (logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])) / 2
        else:
            logits = model(x)
            logits = (logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])) / 2
        out[s:s + BATCH] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float16)
        done = min(s + BATCH, len(u8))
        if (s // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"    {tag} {done}/{len(u8)} | {el:.0f}s | ETA {el/done*(len(u8)-done):.0f}s", flush=True)
    return out


def val_gt(va_idx):
    gt = np.zeros((len(va_idx), 4, H, W), dtype=np.uint8)
    row_of = {str(iid): k for k, iid in enumerate(X[va_idx])}
    for iid, grp in df[df["image_id"].isin(X[va_idx])].groupby("image_id"):
        r = row_of[str(iid)]
        for row in grp.itertuples(index=False):
            if pd.isna(row.mask_rle) or not str(row.mask_rle).strip():
                continue
            c = DEFECT_TYPES.index(row.defect_type)
            gt[r, c] = np.maximum(gt[r, c], rle_decode(row.mask_rle, H, W))
    return gt


val_probs, val_gts, per_fold_thr = [], [], []
for fold, path in sorted(model_paths.items()):
    va_idx = folds[fold - 1][1]
    print(f"STAGE fold {fold} val | {len(va_idx)} imgs | {time.time()-t_start:.0f}s", flush=True)
    model = torch.load(path, map_location=device, weights_only=False).to(device).eval()
    vp = probs_tta(model, train_u8[va_idx], tag=f"val{fold}")
    gt = val_gt(va_idx)
    s_tuned = competition_score(postprocess(vp, TUNED["thresholds"], TUNED["min_component"], TUNED["min_pixels"]), gt)
    print(f"fold {fold} val score (TUNED thr+CC) = {s_tuned:.4f}", flush=True)
    thr = best_thresholds(vp, gt)
    print(f"fold {fold} best thr {thr}", flush=True)
    per_fold_thr.append(thr)
    val_probs.append(vp)
    val_gts.append(gt)
    del model
    gc.collect(); torch.cuda.empty_cache()

gc.collect()
print(f"STAGE 2/5 done | {time.time()-t_start:.0f}s", flush=True)
''')

code(r'''
# ===== STAGE 3/5: threshold choice (LB history says low thr wins; val prefers foldavg, ignore it) =====
for name, thr in {"sweep_tta": TUNED["thresholds"],
                  "foldavg": [float(t) for t in np.mean(np.array(per_fold_thr), axis=0)]}.items():
    scores = [competition_score(postprocess(vp, thr, TUNED["min_component"], TUNED["min_pixels"]), gt)
              for vp, gt in zip(val_probs, val_gts)]
    print(f"candidate {name:9s} thr={[round(t, 3) for t in thr]} | folds {[round(s, 4) for s in scores]} | mean {float(np.mean(scores)):.4f}", flush=True)

BEST_THR = list(TUNED["thresholds"])
print(f"USING sweep_tta (LB-proven) thr={BEST_THR} | {time.time()-t_start:.0f}s", flush=True)

del val_probs, val_gts
gc.collect()
''')

code(r'''
# ===== STAGE 4/5: test probs, 3-model ensemble (TTA) =====
sum_probs = None
for fold, path in sorted(model_paths.items()):
    print(f"STAGE fold {fold} test | {len(test_u8)} imgs | {time.time()-t_start:.0f}s", flush=True)
    model = torch.load(path, map_location=device, weights_only=False).to(device).eval()
    tp = probs_tta(model, test_u8, tag=f"test{fold}")
    np.save(f"/kaggle/working/test_probs_fold{fold}.npy", tp)  # enables free CPU threshold/weight variants
    print(f"fold {fold} probs saved | {time.time()-t_start:.0f}s", flush=True)
    if sum_probs is None:
        sum_probs = tp
    else:
        for i0 in range(0, len(sum_probs), 256):
            sum_probs[i0:i0 + 256] += tp[i0:i0 + 256]
        del tp
    del model
    gc.collect(); torch.cuda.empty_cache()
    print(f"fold {fold} test done | {time.time()-t_start:.0f}s", flush=True)

del train_u8
gc.collect()
print(f"STAGE 4/5 done | {time.time()-t_start:.0f}s", flush=True)
''')

code(r'''
# ===== STAGE 5/5: chunked postprocess -> RLE -> submission =====
n_models = len(model_paths)
rle_dict = {}
for i0 in range(0, len(test_ids), 256):
    chunk = (sum_probs[i0:i0 + 256] / np.float16(n_models)).astype(np.float32)
    pred = postprocess(chunk, BEST_THR, TUNED["min_component"], TUNED["min_pixels"])
    for b in range(pred.shape[0]):
        img_id = str(test_ids[i0 + b])
        for c, dt in enumerate(DEFECT_TYPES):
            m = pred[b, c]
            rle_dict[(img_id, dt)] = rle_encode(m) if m.sum() > 0 else " "
    if (i0 // 256) % 4 == 0:
        print(f"  postproc {min(i0+256, len(test_ids))}/{len(test_ids)} | {time.time()-t_start:.0f}s", flush=True)
del sum_probs
gc.collect()

sub = sub_raw.copy()
sub["mask_rle"] = [rle_dict.get((str(r.image_id), str(r.defect_type)), " ") for r in sub.itertuples(index=False)]
sub = sub[list(sub_raw.columns)]
sub.to_csv("/kaggle/working/submission.csv", index=False)

non_empty = (sub["mask_rle"] != " ").sum()
print(f"\nsubmission.csv | rows {len(sub)} | masks {non_empty} ({non_empty/len(sub):.2%})")

chk = pd.read_csv("/kaggle/working/submission.csv")
assert list(chk.columns) == list(sub_raw.columns), "column mismatch"
assert len(chk) == len(sub_raw), f"row count {len(chk)} != {len(sub_raw)}"
assert set(chk["image_id"]) == set(sub_raw["image_id"]), "image_id mismatch"
assert chk["mask_rle"].notna().all(), "NaN in mask_rle"
assert chk["mask_rle"].str.match(r"^(\s*|\d+( \d+)*)$").all(), "malformed RLE"
print(f"VALIDATION PASSED | total {time.time()-t_start:.0f}s | thresholds {BEST_THR}", flush=True)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "infer-v2.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "coder057/dl-infer-v2", "title": "DL-infer-v2", "code_file": "infer-v2.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
        "dataset_sources": ["coder057/fold1-model", "coder057/folds23-models"],
        "kernel_sources": [], "competition_sources": ["industrial-surface-defect-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("written:", OUT / "infer-v2.ipynb")
