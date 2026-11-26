"""Shared notebook patcher: make the competition root resilient to Kaggle's shifting mount path.

fold4 failed twice with `FileNotFoundError: .../public/train.csv` while fold1/2/3 used the same
literal path, so the mount root is not stable across sessions. Resolve it by searching instead.
"""

OLD = 'ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")'


def fix_root(src, need):
    """Replace the hardcoded ROOT assignment with one that falls back to a filesystem search."""
    new = f'''{OLD}
if not (ROOT / "{need}").exists():  # Kaggle mounts the competition data at varying depths
    _hits = sorted(Path("/kaggle/input").rglob("{need}"))
    if _hits:
        ROOT = _hits[0].parent
print("ROOT:", ROOT, "| has {need}:", (ROOT / "{need}").exists(), flush=True)
assert (ROOT / "{need}").exists(), f"competition data not mounted: {{ROOT}}"'''
    assert OLD in src, "ROOT line not found - source notebook changed"
    return src.replace(OLD, new, 1)
