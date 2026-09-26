# Failure injection for the D2 check: INJECT_FAIL=1 PYTHONPATH=.agent/cache/tools/inject_fail python main.py test_sim ...
# Raises on day 2 for every ALNS policy (works in spawned parallel workers too); test_sim must exit non-zero.
import importlib.abc, importlib.util, os, sys
TARGET = "logic.src.pipeline.simulations.states.running"

def _patch(mod):
    orig = mod.run_day
    def run_day(dc):
        if getattr(dc, "day", None) == 2 and "alns" in str(getattr(dc, "policy_name", "")):
            raise RuntimeError("injected day-2 failure")
        return orig(dc)
    mod.run_day = run_day

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET: return None
        sys.meta_path.remove(self)
        try: spec = importlib.util.find_spec(name)
        finally: sys.meta_path.insert(0, self)
        loader = spec.loader; exec_module = loader.exec_module
        def exec_and_patch(module):
            exec_module(module); _patch(module)
        loader.exec_module = exec_and_patch
        return spec

if os.environ.get("INJECT_FAIL") == "1":
    sys.meta_path.insert(0, _Finder())
