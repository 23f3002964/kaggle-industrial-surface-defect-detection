#!/usr/bin/env python3
"""Generate infer-v3: baseline (Kaggle Model, LB 0.62361) test probs only -> npy + sanity submission.
No training data needed; ~3 min GPU. Its output probs feed post-v2 (CPU) combinations.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "infer-v3"
OUT.mkdir(exist_ok=True)

cells = []


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s})


code(r'''
import time, gc
import numpy as np, pandas as pd, cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F

t_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| GPUs:", torch.cuda.device_count(), flush=True)
ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
TEST_DIR, SAMPLE = ROOT / "test", ROOT / "sample_submission.csv"
DEFECT_TYPES = ["defect_a", "defect_b", "defect_c", "defect_d"]
TUNED = {"thresholds": [0.175, 0.05, 0.375, 0.2],
         "min_component": [48, 32, 160, 256], "min_pixels": [64, 0, 192, 384]}
BATCH = 64
''')

code(r'''
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
# ===== baseline model (sweep-v1 source, LB 0.62361) =====
cands = sorted(Path("/kaggle/input").rglob("*.pth"))
print("model files:", [str(p) for p in cands], flush=True)
assert cands, "no .pth attached"
model = torch.load(cands[0], map_location=device, weights_only=False).to(device).eval()
print(f"loaded {cands[0]} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)

sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()
H, W = 128, 800

test_u8 = np.empty((len(test_ids), H, W), dtype=np.uint8)
for i, iid in enumerate(tqdm(test_ids, desc="test imgs", dynamic_ncols=True)):
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TEST_DIR / name) as im:
        test_u8[i] = np.asarray(im.convert("L"), dtype=np.uint8)
print(f"test_u8 {test_u8.nbytes/1e9:.2f}GB | {time.time()-t_start:.0f}s", flush=True)


@torch.inference_mode()
def probs_tta(model, u8, tag=""):
    out = np.empty((len(u8), 4, H, W), dtype=np.float16)
    t0 = time.time()
    for s in range(0, len(u8), BATCH):
        x = torch.from_numpy(u8[s:s + BATCH]).unsqueeze(1).float().div_(255).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            logits = (logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])) / 2
        out[s:s + BATCH] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float16)
        done = min(s + BATCH, len(u8))
        if (s // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"  {tag} {done}/{len(u8)} | {el:.0f}s | ETA {el/done*(len(u8)-done):.0f}s", flush=True)
    return out


probs = probs_tta(model, test_u8, tag="baseline")
np.save("/kaggle/working/test_probs_baseline.npy", probs)
print(f"saved test_probs_baseline.npy | {time.time()-t_start:.0f}s", flush=True)

pred = postprocess(probs, TUNED["thresholds"], TUNED["min_component"], TUNED["min_pixels"])
rle_dict = {}
for b in range(pred.shape[0]):
    img_id = str(test_ids[b])
    for c, dt in enumerate(DEFECT_TYPES):
        m = pred[b, c]
        rle_dict[(img_id, dt)] = rle_encode(m) if m.sum() > 0 else " "

sub = sub_raw.copy()
sub["mask_rle"] = [rle_dict.get((str(r.image_id), str(r.defect_type)), " ") for r in sub.itertuples(index=False)]
sub = sub[list(sub_raw.columns)]
sub.to_csv("/kaggle/working/submission.csv", index=False)
non_empty = int((sub["mask_rle"] != " ").sum())
print(f"submission.csv | rows {len(sub)} | masks {non_empty} ({non_empty/len(sub):.2%})", flush=True)

chk = pd.read_csv("/kaggle/working/submission.csv")
assert list(chk.columns) == list(sub_raw.columns)
assert len(chk) == len(sub_raw)
assert set(chk["image_id"]) == set(sub_raw["image_id"])
assert chk["mask_rle"].notna().all()
assert chk["mask_rle"].str.match(r"^(\s*|\d+( \d+)*)$").all()
print(f"VALIDATION PASSED | total {time.time()-t_start:.0f}s", flush=True)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "infer-v3.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "coder057/dl-infer-v3", "title": "DL-infer-v3", "code_file": "infer-v3.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [], "kernel_sources": [],
        "competition_sources": ["industrial-surface-defect-detection"],
        "model_sources": ["coder057/attention-unet/pytorch/baseline-v1.0/1"],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("written:", OUT / "infer-v3.ipynb")
