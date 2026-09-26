import sys, os; sys.path.insert(0, os.getcwd())  # run from the repo root: .venv/bin/python .agent/cache/tools/import_sweep.py
import importlib, pathlib, sys, traceback, warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_MODE", "disabled")
root = pathlib.Path("logic")
mods = []
for p in sorted(root.rglob("*.py")):
    if "__pycache__" in p.parts or "wsmart_bin_analysis" in p.parts: continue
    parts = list(p.with_suffix("").parts)
    if parts[-1] == "__init__": parts = parts[:-1]
    mods.append(".".join(parts))
fails = {}
for m in mods:
    try:
        importlib.import_module(m)
    except BaseException as e:  # noqa
        tb = traceback.format_exc().strip().splitlines()
        fails[m] = f"{type(e).__name__}: {e}"
print(f"{len(mods)} modules, {len(fails)} failed")
for m, e in fails.items():
    print(f"FAIL {m}\n     {e[:300]}")
