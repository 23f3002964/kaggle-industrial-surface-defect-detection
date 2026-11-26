import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DS = sys.argv[1] if len(sys.argv) > 1 else "coder057/isdd-csvs-nov16b"
SLUG = "isdd-csvs-copy"
ACCT = sys.argv[2] if len(sys.argv) > 2 else "abdulmajid786123"
D = HERE / f"acct-{ACCT}-{SLUG}"
D.mkdir(exist_ok=True)

CODE = '''import time
import pandas as pd
from pathlib import Path

t0 = time.time()
ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
if not (ROOT / "sample_submission.csv").exists():
    _hits = sorted(Path("/kaggle/input").rglob("sample_submission.csv"))
    if _hits: ROOT = _hits[0].parent
assert (ROOT / "sample_submission.csv").exists(), f"competition data not mounted: {ROOT}"
sample = pd.read_csv(ROOT / "sample_submission.csv")
assert list(sample.columns) == ["ID", "image_id", "defect_type", "mask_rle"], list(sample.columns)
print(f"sample {len(sample)} rows", flush=True)

srcs = sorted(Path("/kaggle/input").rglob("submission_v*.csv"))
print("csvs:", [p.name for p in srcs], flush=True)
assert srcs, "csv dataset not mounted"
for p in srcs:
    df = pd.read_csv(p)
    assert list(df.columns) == list(sample.columns), f"{p.name}: {list(df.columns)}"
    for col in ("ID", "image_id", "defect_type"):
        assert df[col].tolist() == sample[col].tolist(), f"{p.name}: {col} mismatch"
    out = Path("/kaggle/working") / p.name
    df.to_csv(out, index=False)
    print(f"{out.name}: {len(df)} rows | {int((df.mask_rle.astype(str).str.strip() != '').sum())} masks", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
'''

nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": SLUG, "metadata": {}, "outputs": [],
                 "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(D / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"{ACCT}/{SLUG}", "title": "isdd csvs copy", "code_file": f"{SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_internet": False, "dataset_sources": [DS],
        "competition_sources": ["industrial-surface-defect-detection"], "kernel_sources": []}
(D / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
compile(CODE, "copy", "exec")
print("written", D, "dataset", DS)
