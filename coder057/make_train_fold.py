#!/usr/bin/env python3
"""Generate a single-fold 32ep training kernel for an arbitrary account.

usage: python3 make_train_fold.py <account> <fold> <epochs>
  e.g. python3 make_train_fold.py bytescode0115 2 32  ->  bytescode0115/dl-train-fold2
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
from nbpatch import fix_root
acct, fold, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
slug = f"dl-train-fold{fold}"
OUT = HERE / f"acct-{acct}-{slug}"
OUT.mkdir(exist_ok=True)

SRC = json.load(open(HERE / "train-v7" / "train-v7.ipynb"))["cells"]
cells = [dict(c) for c in SRC[:9]]
cells[1]["source"] = fix_root("".join(cells[1]["source"]), "train.csv")

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": f'''# ===== train fold {fold} only, {epochs} epochs, single GPU =====
t_start = time.time()
EPOCHS = {epochs}
kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
fold_splits = list(kfold.split(X, y))
base_train = SurfaceDefectDataset(images_u8, masks, is_train=True)
base_val = SurfaceDefectDataset(images_u8, masks, is_train=False)
val_scores = {{}}

for fold in ({fold},):
    tr_idx, va_idx = fold_splits[fold - 1]
    print(f"STAGE fold {{fold}}/5 | train {{len(tr_idx)}} val {{len(va_idx)}} | {{time.time()-t_start:.0f}}s", flush=True)
    tr_loader = DataLoader(Subset(base_train, tr_idx), batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(Subset(base_val, va_idx), batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = AttentionDefectUNet(in_channels=1, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = SoftDiceBCELoss(bce_weight=0.4, dice_weight=0.6)
    save_path = f"/kaggle/working/ATTENTION-UNET-FOLD{{fold}}.pth"
    history = fit_model(model, tr_loader, va_loader, optimizer, criterion,
                        epochs=EPOCHS, log_every=40, run_name=f"ATTN-FOLD{{fold}}", save_path=save_path)
    val_scores[fold] = max(history["valid"]["score"])
    print(f"fold {{fold}} best val {{val_scores[fold]:.4f}} | {{time.time()-t_start:.0f}}s elapsed", flush=True)
    json.dump(val_scores, open("/kaggle/working/fold_scores.json", "w"))

print("TRAINING DONE:", val_scores, flush=True)
''',
})

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": f"{acct}/{slug}", "title": f"DL-train-fold{fold}", "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
        "dataset_sources": [], "kernel_sources": [],
        "competition_sources": ["industrial-surface-defect-detection"],
        "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"written {OUT} | id {meta['id']} | fold {fold} ep {epochs}")
