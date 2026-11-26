#!/usr/bin/env python3
"""Create e2e kernel variants with different per-class targets."""
import json
from pathlib import Path

HERE = Path(__file__).parent
INFER = json.load(open(HERE / "infer-v3" / "infer-v3.ipynb"))["cells"]
POST = (HERE / "make_postv.py").read_text().split("NB_HEAD = r'''", 1)[1].split("\n'''", 1)[0]

# v81 best: defect_b 492, defect_c 1468, defect_a 670, defect_d 1219 = 3849 masks, 0.65677
# Target: defect_b 450-520, defect_c 1450-1500, defect_a 620-670, defect_d 1150-1220

variants = [
    (85, {"defect_a": 670, "defect_b": 450, "defect_c": 1468, "defect_d": 1219}),
    (86, {"defect_a": 670, "defect_b": 470, "defect_c": 1468, "defect_d": 1219}),
    (87, {"defect_a": 670, "defect_b": 490, "defect_c": 1468, "defect_d": 1219}),
    (88, {"defect_a": 670, "defect_b": 510, "defect_c": 1468, "defect_d": 1219}),
    (89, {"defect_a": 670, "defect_b": 520, "defect_c": 1468, "defect_d": 1219}),
    (90, {"defect_a": 650, "defect_b": 492, "defect_c": 1468, "defect_d": 1219}),
    (91, {"defect_a": 620, "defect_b": 492, "defect_c": 1468, "defect_d": 1219}),
    (92, {"defect_a": 670, "defect_b": 492, "defect_c": 1450, "defect_d": 1219}),
    (93, {"defect_a": 670, "defect_b": 492, "defect_c": 1480, "defect_d": 1219}),
    (94, {"defect_a": 670, "defect_b": 492, "defect_c": 1500, "defect_d": 1219}),
    (95, {"defect_a": 670, "defect_b": 492, "defect_c": 1468, "defect_d": 1150}),
    (96, {"defect_a": 670, "defect_b": 492, "defect_c": 1468, "defect_d": 1180}),
]

for vnum, targets in variants:
    SETTINGS = {
        "weights": {"baseline": 1.0, "ft1b": 1.0, "ft3b": 1.0, "ft4b": 1.0, "ft5b": 1.0},
        "mode": "mean",
        "target_masks": targets,
        "min_component": [48, 32, 160, 256],
        "min_pixels": [64, 0, 192, 384],
    }
    POST_FILLED = POST.replace("__SETTINGS__", json.dumps(SETTINGS))
    POST_FILLED = POST_FILLED.replace(
        'for p in Path("/kaggle/input").rglob("test_probs_*.npy"):',
        'for p in [q for root in ("/kaggle/input", "/kaggle/working") for q in Path(root).rglob("test_probs_*.npy")]:')

    LOAD = '''# ===== stage 1/3 | resolve the 5 checkpoints and load the test images =====
CKPTS = {}
for p in sorted(Path("/kaggle/input").rglob("*.pth")):
    if "FTB" not in p.name:
        tag = "baseline"
    else:
        fold_num = p.name.split("FOLD")[1].split("FTB")[0]
        tag = f"ft{fold_num}b"
    CKPTS[tag] = p
print("checkpoints:", {k: str(v) for k, v in CKPTS.items()}, flush=True)
assert set(CKPTS) == {"baseline", "ft1b", "ft3b", "ft4b", "ft5b"}, f"unexpected members {sorted(CKPTS)}"

sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()
H, W = 128, 800
test_u8 = np.empty((len(test_ids), H, W), dtype=np.uint8)
for i, iid in enumerate(tqdm(test_ids, desc="test imgs", dynamic_ncols=True)):
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TEST_DIR / name) as im:
        test_u8[i] = np.asarray(im.convert("L"), dtype=np.uint8)
print(f"test_u8 {test_u8.nbytes/1e9:.2f}GB | {len(test_ids)} imgs | {time.time()-t_start:.0f}s", flush=True)
'''

    INFER_ALL = '''# ===== stage 2/3 | hflip-TTA inference for every blend member, uint8 dumps =====
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


for i, (tag, ckpt) in enumerate(sorted(CKPTS.items()), 1):
    model = torch.load(ckpt, map_location=device, weights_only=False).to(device).eval()
    print(f"\\n[{i}/5] {tag} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | "
          f"{time.time()-t_start:.0f}s elapsed", flush=True)
    tp = probs_tta(model, test_u8, tag=tag)
    u8 = np.clip(tp * 255.0, 0, 255).round().astype(np.uint8)
    np.save(f"/kaggle/working/test_probs_{tag}.npy", u8)
    print(f"[{i}/5] {tag} saved uint8 | mean {float(tp.mean()):.4f} | {time.time()-t_start:.0f}s", flush=True)
    del model, tp, u8
    gc.collect(); torch.cuda.empty_cache()
'''

    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        f"# E2E variant v{vnum} — per-class targets {targets}\n",
        "\n",
        "End-to-end: 5 checkpoints -> hflip-TTA -> blend -> calibrated masks.\n"]}]
    for c in INFER[:2]:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": "".join(c["source"])})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": LOAD})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": INFER_ALL})
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 3/3 - blend and calibrated masks\n"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": POST_FILLED})

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"acct-coder057-isdd-e2e-v{vnum}"
    out.mkdir(exist_ok=True)
    name = f"isdd-e2e-v{vnum}"
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"coder057/{name}", "title": f"ISDD-E2E-V{vnum}", "code_file": f"{name}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": False, "enable_gpu": True,
            "enable_internet": False, "keywords": ["segmentation", "unet", "tta"],
            "dataset_sources": ["coder057/isdd-ftb-models"],
            "kernel_sources": [], "competition_sources": ["industrial-surface-defect-detection"],
            "model_sources": ["coder057/attention-unet/pytorch/baseline-v1.0/1"],
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"written {out} | targets {list(targets.values())} total {sum(targets.values())}")

print("done")
