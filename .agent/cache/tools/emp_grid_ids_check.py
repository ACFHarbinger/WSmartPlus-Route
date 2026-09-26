"""DS-13 check: the empirical fill grid must describe the same bins (same IDs, same order)
as the routed graph built from the focus graph.

usage (repo root): .venv/bin/python .agent/cache/tools/emp_grid_ids_check.py [area] [focus_graph]
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from logic.src.constants.paths import ROOT_DIR
from logic.src.data.processor import process_data
from logic.src.data.processor.setup import setup_basedata
from logic.src.pipeline.simulations.bins import Bins
from logic.src.pipeline.simulations.repository import FileSystemRepository, load_indices, set_repository

area = sys.argv[1] if len(sys.argv) > 1 else "riomaior"
focus = sys.argv[2] if len(sys.argv) > 2 else "graphs_20V_1N_plastic.json"
dd = os.path.join(ROOT_DIR, "data", "simulator")
set_repository(FileSystemRepository(dd))
data, coords, depot = setup_basedata(20, dd, area, "plastic")
idx = load_indices(focus, 1, 20, len(data))[0]
_, routed = process_data(data, coords, depot, idx)
routed_ids = [str(int(i)) for i in routed["ID"].tolist()[1:]]  # row 0 is the depot
from logic.src.pipeline.simulations.states.initializing import routed_bin_ids  # noqa: E402
from logic.src.utils.data.loader import load_grid_base  # noqa: E402

grid = load_grid_base(None, area, dd, ids=routed_bin_ids(routed))
bins = Bins(20, dd, "emp", area=area, waste_type="plastic", grid=grid, n_days=2, n_samples=1)
grid_ids = [str(c) for c in bins.grid.data.columns]
assert len(bins.dist_param1) == 20, len(bins.dist_param1)
print("routed IDs[:6]", routed_ids[:6])
print("grid   IDs[:6]", grid_ids[:6])
assert grid_ids == routed_ids, f"overlap {len(set(grid_ids) & set(routed_ids))}/{len(routed_ids)}"
print("EMP GRID MATCHES ROUTED BINS")
