#!/usr/bin/env python3
"""Generate train-v6: training ONLY for folds 4 and 5 (26 epochs each).
Reuses model/loss/fit cells verbatim from train-v1.ipynb (no drift), but:
  - uint8 image preload (1.2GB instead of 4.7GB)
  - NO test inference / NO threshold sweep inside the training kernel (that caused the v5 OOM)
  - saves ATTENTION-UNET-FOLD{4,5}.pth + val scores json
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "train-v6"
OUT.mkdir(exist_ok=True)
SRC = json.load(open(HERE / "train-v1" / "train-v1.ipynb"))["cells"]
src = lambda i: "".join(SRC[i]["source"])

cells = []


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s})


code(src(1).replace("segmentation-models-pytorch", "iterative-stratification"))

code(r'''
import os, gc, time, json, random
import numpy as np, pandas as pd, cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from contextlib import nullcontext
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def seed_everything(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
print(f"Device: {device} | GPUs: {num_gpus}", flush=True)

ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
TRAIN_DIR, CSV_PATH = ROOT / "train", ROOT / "train.csv"
DEFECT_TYPES = ["defect_a", "defect_b", "defect_c", "defect_d"]
CLASS_TO_IDX = {n: i for i, n in enumerate(DEFECT_TYPES)}
CONFIG = {"seed": 42, "lr": 1e-3, "weight_decay": 1e-4,
          "class_thresholds": [0.20, 0.35, 0.575, 0.40],
          "min_defect_pixels": 255, "min_component_pixels": 75}
''')

code(src(10))     # rle_decode / rle_encode
code(r'''
df = pd.read_csv(CSV_PATH)
image_df = pd.DataFrame({"image_id": df["image_id"].drop_duplicates().values})
for d in DEFECT_TYPES:
    pos = df.loc[(df["defect_type"] == d) & df["mask_rle"].notna()
                 & (df["mask_rle"].astype(str).str.strip() != ""), "image_id"].unique()
    image_df[d] = image_df["image_id"].isin(pos).astype(int)
X = image_df["image_id"].values
y = image_df[DEFECT_TYPES].values
print(f"Images: {len(X):,} | positive rate per class: {y.mean(0).round(3)}", flush=True)
''')
code(src(27))     # DoubleConv / DownBlock / UpBlock / DefectUNet
code(src(29))     # AttentionGate / AttentionUpBlock / AttentionDefectUNet
code(src(37))     # SoftDiceBCELoss + metrics
code(src(39))     # get_amp / train_one_epoch / validate / fit_model

code(r'''
# ===== uint8 preload (1.2GB images instead of 4.7GB float32) =====
df = pd.read_csv(CSV_PATH)
H, W = int(df["height"].iloc[0]), int(df["width"].iloc[0])
n = len(X)
images_u8 = np.empty((n, H, W), dtype=np.uint8)
masks = np.zeros((n, 4, H, W), dtype=np.uint8)
groups = {str(i): g for i, g in df.groupby("image_id")}
for i, image_id in enumerate(tqdm(X, desc="preload", dynamic_ncols=True)):
    iid = str(image_id)
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TRAIN_DIR / name) as im:
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    if a.shape != (H, W):
        raise ValueError(f"{iid} shape {a.shape}")
    images_u8[i] = a
    g = groups.get(iid)
    if g is not None:
        for row in g.itertuples(index=False):
            if pd.isna(row.mask_rle) or not str(row.mask_rle).strip():
                continue
            c = CLASS_TO_IDX[row.defect_type]
            masks[i, c] = np.maximum(masks[i, c], rle_decode(row.mask_rle, H, W))
print(f"preload done: images {images_u8.nbytes/1e9:.2f}GB | masks {masks.nbytes/1e9:.2f}GB", flush=True)
''' + "\n" + src(19).split("class SurfaceDefectDataset")[0] + r'''
class SurfaceDefectDataset(Dataset):
    def __init__(self, images_u8, masks, is_train=False):
        self.images, self.masks, self.is_train = images_u8, masks, is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        image = image[None]  # (1,H,W)
        mask = self.masks[idx]
        if self.is_train:
            image, mask = augment(image, mask)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(mask.astype(np.float32))
''')

code(r'''
# ===== train folds 4 and 5 only (no test inference -> stays inside RAM) =====
t_start = time.time()
TIME_BUDGET = 5.9 * 3600
EPOCHS = 26
kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
fold_splits = list(kfold.split(X, y))
base_train = SurfaceDefectDataset(images_u8, masks, is_train=True)
base_val = SurfaceDefectDataset(images_u8, masks, is_train=False)
val_scores = {}

for fold in (4, 5):
    if time.time() - t_start > TIME_BUDGET:
        print(f"TIME BUDGET hit before fold {fold}", flush=True)
        break
    tr_idx, va_idx = fold_splits[fold - 1]
    print(f"STAGE fold {fold}/5 | train {len(tr_idx)} val {len(va_idx)} | {time.time()-t_start:.0f}s", flush=True)
    tr_loader = DataLoader(Subset(base_train, tr_idx), batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(Subset(base_val, va_idx), batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = AttentionDefectUNet(in_channels=1, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = SoftDiceBCELoss(bce_weight=0.4, dice_weight=0.6)
    save_path = f"/kaggle/working/ATTENTION-UNET-FOLD{fold}.pth"
    history = fit_model(model, tr_loader, va_loader, optimizer, criterion,
                        epochs=EPOCHS, log_every=40, run_name=f"ATTN-FOLD{fold}", save_path=save_path)
    val_scores[fold] = max(history["valid"]["score"])
    print(f"fold {fold} best val {val_scores[fold]:.4f} | {time.time()-t_start:.0f}s elapsed", flush=True)
    json.dump(val_scores, open("/kaggle/working/fold_scores.json", "w"))
    del model, optimizer, tr_loader, va_loader
    gc.collect(); torch.cuda.empty_cache()

print("TRAINING DONE:", val_scores, flush=True)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "train-v6.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "coder057/dl-train-v6", "title": "DL-train-v6", "code_file": "train-v6.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
        "dataset_sources": [], "kernel_sources": [],
        "competition_sources": ["industrial-surface-defect-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("written:", OUT / "train-v6.ipynb")
