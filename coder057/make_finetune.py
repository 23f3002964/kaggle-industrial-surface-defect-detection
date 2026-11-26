#!/usr/bin/env python3
"""Generate a fine-tune kernel: load fold N checkpoint, train K more epochs, emit test probs.

usage: python3 make_finetune.py <account> <fold> <src_kernel> <epochs>
  e.g. python3 make_finetune.py coder057 4 coder057/dl-train-v7 7
Reuses the train-v9 notebook (fold1ft) layout: same splits (seed 42), lr 3e-4.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
acct, fold, src_kernel, epochs = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
slug = f"dl-ft-fold{fold}"
OUT = HERE / f"acct-{acct}-{slug}"
OUT.mkdir(exist_ok=True)

SRC = json.load(open(HERE / "train-v9" / "train-v9.ipynb"))["cells"]
cells = [dict(c) for c in SRC[:9]]
cell = ''.join(SRC[9]['source'])

cell = cell.replace('FINETUNE_FROM = next(Path("/kaggle/input").rglob("ATTENTION-UNET-FOLD1.pth"))',
                    f'FINETUNE_FROM = next(Path("/kaggle/input").rglob("ATTENTION-UNET-FOLD{fold}.pth"))')
cell = cell.replace('EPOCHS = 12', f'EPOCHS = {epochs}')
cell = cell.replace('for fold in (1,):', f'for fold in ({fold},):')
cell = cell.replace('ATTENTION-UNET-FOLD1FT.pth', f'ATTENTION-UNET-FOLD{fold}FT.pth')
cell = cell.replace('test_probs_fold1.npy', f'test_probs_fold{fold}.npy')
cell = cell.replace('fold1ft probs saved', f'fold{fold}ft probs saved')
cell = cell.replace('val_scores[f"fold{fold}ft"]', "val_scores['ft']")
cell = cell.replace("val_scores[f'fold{fold}ft']", "val_scores['ft']")
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell})

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": f"{acct}/{slug}", "title": f"DL-ft-fold{fold}", "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
        "dataset_sources": [], "kernel_sources": [src_kernel],
        "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

check = ''.join(nb['cells'][9]['source'])
errs = []
for i, c in enumerate(nb['cells'][1:], 1):
    try:
        compile(''.join(c['source']), f"c{i}", "exec")
    except SyntaxError as e:
        errs.append((i, e.lineno, e.msg))
print(f"written {OUT} | id {meta['id']} | src {src_kernel} | fold {fold} | ep {epochs}")
print("checks:", errs, f'FOLD{fold}.pth' in check, f'for fold in ({fold},):' in check,
      f'test_probs_fold{fold}.npy' in check, f'EPOCHS = {epochs}' in check)
