#!/usr/bin/env python3
"""Reverse share: run inference on coder057's fine-tuned checkpoints (uploaded as a dataset to
bytescode0115) and build 2 postproc variants that include those probs.

usage: python3 make_ft_probs_acct.py <account>      (dataset must be <account>/coder057-ft-models)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
from nbpatch import fix_root
acct = sys.argv[1]

# ---------- inference kernel: checkpoint -> probs (tags ft1/ft4) ----------
SRC = json.load(open(HERE / "infer-v4" / "infer-v4.ipynb"))["cells"]
cells = [dict(SRC[0]), dict(SRC[1])]
cells[0]["source"] = fix_root("".join(cells[0]["source"]), "sample_submission.csv")
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": '''sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()
H, W = 128, 800

test_u8 = np.empty((len(test_ids), H, W), dtype=np.uint8)
for i, iid in enumerate(tqdm(test_ids, desc="test imgs", dynamic_ncols=True)):
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TEST_DIR / name) as im:
        test_u8[i] = np.asarray(im.convert("L"), dtype=np.uint8)
print(f"test_u8 {test_u8.nbytes/1e9:.2f}GB | {time.time()-t_start:.0f}s", flush=True)

pths = sorted(Path("/kaggle/input").rglob("ATTENTION-UNET-FOLD*FT.pth"))
print("checkpoints:", [str(p) for p in pths], flush=True)
assert pths, "no FT checkpoints found"
jobs = [(f"ft{p.stem.split('FOLD')[-1].replace('FT','')}", p) for p in pths]
print("jobs:", jobs, flush=True)


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
        el = time.time() - t0
        print(f"  {tag} {min(s+BATCH, len(u8))}/{len(u8)} | {el:.0f}s", flush=True)
    return out


for tag, pth in jobs:
    model = torch.load(pth, map_location=device, weights_only=False).to(device).eval()
    tp = probs_tta(model, test_u8, tag=tag)
    np.save(f"/kaggle/working/test_probs_{tag}.npy", tp)
    print(f"{tag} saved | mean {tp.mean():.4f} | {time.time()-t_start:.0f}s", flush=True)
    del model, tp
    gc.collect(); torch.cuda.empty_cache()

print(f"DONE | total {time.time()-t_start:.0f}s", flush=True)
'''})

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / f"acct-{acct}-dl-probs-ft"
out.mkdir(exist_ok=True)
(out / "dl-probs-ft.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"{acct}/dl-probs-ft", "title": "DL-probs-ft", "code_file": "dl-probs-ft.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [f"{acct}/coder057-ft-models"], "kernel_sources": [],
        "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"written {out} | id {meta['id']} | dataset {meta['dataset_sources']}")

# ---------- postproc variants that use ft1/ft4 on top of the 4 folds ----------
src = (HERE / "make_postv.py").read_text()
NB_HEAD = src.split("NB_HEAD = r'''", 1)[1].split("\n'''", 1)[0]
BASE_TARGET = {"defect_a": 675, "defect_b": 570, "defect_c": 1440, "defect_d": 1271}
FOLDS = {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}
VARIANTS = [
    dict(name="acct-f", weights=dict(FOLDS, ft1=1.0, ft4=1.0), scale=1.0),
    dict(name="acct-g", weights=dict(FOLDS, ft1=1.0, ft4=1.0), scale=0.95, mc_scale=0.55),
]
for v in VARIANTS:
    slug = f"dl-post-{v['name']}"
    d = HERE / f"acct-{acct}-{slug}"
    d.mkdir(exist_ok=True)
    mc = v.get("mc_scale", 1.0)
    settings = {"weights": v["weights"], "mode": "mean",
                "target_masks": {k: int(round(n * v["scale"])) for k, n in BASE_TARGET.items()},
                "min_component": [int(round(x * mc)) for x in (48, 32, 160, 256)],
                "min_pixels": [int(round(x * mc)) for x in (64, 0, 192, 384)]}
    cell = NB_HEAD.replace("__SETTINGS__", json.dumps(settings))
    nb2 = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell}],
           "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
    (d / f"{slug}.ipynb").write_text(json.dumps(nb2, indent=1))
    m2 = {"id": f"{acct}/{slug}", "title": f"DL-post-{v['name']}", "code_file": f"{slug}.ipynb",
          "language": "python", "kernel_type": "notebook", "is_private": True,
          "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
          "dataset_sources": [], "kernel_sources": [f"{acct}/dl-infer-all", f"{acct}/dl-probs-ft"],
          "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(m2, indent=2))
    print(f"  {v['name']}: {list(v['weights'])} masks {sum(settings['target_masks'].values())}")

errs = []
for i, c in enumerate(nb["cells"], 1):
    try:
        compile("".join(c["source"]), f"c{i}", "exec")
    except SyntaxError as e:
        errs.append((i, e.lineno, e.msg))
print("inference syntax:", errs)
