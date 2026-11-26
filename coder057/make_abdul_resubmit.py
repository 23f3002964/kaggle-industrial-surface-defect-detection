import json
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "acct-abdulmajid786123-csv-resubmit"
D.mkdir(exist_ok=True)

CODE = '''import time
import pandas as pd
from pathlib import Path

t0 = time.time()
ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
if not (ROOT / "sample_submission.csv").exists():
    _hits = sorted(Path("/kaggle/input").rglob("sample_submission.csv"))
    if _hits: ROOT = _hits[0].parent
print("ROOT:", ROOT, flush=True)
assert (ROOT / "sample_submission.csv").exists(), f"competition data not mounted: {ROOT}"

sample = pd.read_csv(ROOT / "sample_submission.csv")
assert list(sample.columns) == ["ID", "image_id", "defect_type", "mask_rle"], list(sample.columns)
n_test = len(list((ROOT / "test").iterdir())) if (ROOT / "test").exists() else -1
print(f"sample {len(sample)} rows | test dir {n_test} files", flush=True)
assert n_test > 0, "competition test data missing"

src = sorted(Path("/kaggle/input").rglob("submission.csv"))
print("dataset csvs:", [str(p) for p in src], flush=True)
assert len(src) == 1, f"expected exactly 1 submission.csv, found {src}"
df = pd.read_csv(src[0])
assert list(df.columns) == list(sample.columns), list(df.columns)
for col in ("ID", "image_id", "defect_type"):
    assert df[col].tolist() == sample[col].tolist(), f"{col} order mismatch"
out = Path("/kaggle/working/submission.csv")
df.to_csv(out, index=False)
print(f"{out}: {len(df)} rows | {int((df.mask_rle.astype(str).str.strip() != '').sum())} masks | {out.stat().st_size/1e6:.2f}MB", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
'''

nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": "csv-resubmit-abdul", "metadata": {},
                 "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(D / "resubmit.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "abdulmajid786123/isdd-csv-resubmit",
        "title": "isdd csv resubmit",
        "code_file": "resubmit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": ["coder057/isdd-best-csv"],
        "competition_sources": ["industrial-surface-defect-detection"],
        "kernel_sources": []}
(D / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
compile(CODE, "resubmit", "exec")
print("written", D)
