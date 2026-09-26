#!/usr/bin/env python3
"""Lane B reproductions against commit 70e660b03.

Run from a worktree of that commit (submodule + data symlinked):

    /home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python \
        /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/grok_lane_b_repro_20260925.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

ROOT = os.environ.get("WSR_ROOT", "/tmp/grok-wsr")
DATA = os.environ.get(
    "WSR_DATA",
    "/home/pkhunter/Repositories/Doc/WSmart-Route/data/simulator",
)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    failures = 0

    section("B-grok-01 empirical bins vs focus-graph bins")
    focus = json.load(open(os.path.join(DATA, "bins_selection/graphs_20V_1N_plastic.json")))[0]
    old_info = pd.read_csv(os.path.join(DATA, "coordinates/old_out_info[riomaior].csv"))
    new_info = pd.read_csv(os.path.join(DATA, "coordinates/out_info[riomaior].csv"))
    # filesystem._get_riomaior_data filters plastic then the simulator iloc's the focus list and sorts by ID
    plastic = old_info[old_info["Tipo de Residuos"] == "Mistura de embalagens"].reset_index(drop=True)
    routed = plastic.iloc[focus].sort_values("ID")
    grid_rows = new_info.iloc[list(range(20))]
    print(f"focus indices[:5]={focus[:5]} routed IDs[:8]={routed['ID'].head(8).tolist()}")
    print(f"grid arange(20) IDs[:8]={grid_rows['ID'].head(8).tolist()}")
    overlap = set(routed["ID"].astype(str)) & set(grid_rows["ID"].astype(str))
    print(f"ID overlap {len(overlap)}/20")
    if len(overlap) == 20:
        print("UNEXPECTED: sets match")
        failures += 1
    fig_info = pd.read_csv(os.path.join(DATA, "coordinates/out_info[figdafoz].csv"))
    fig_plastic = fig_info[fig_info["description"] == "Mistura de embalagens"].reset_index(drop=True)
    fig_routed = fig_plastic.iloc[focus].sort_values("ID")
    fig_grid = fig_info.iloc[list(range(20))]
    fig_overlap = set(fig_routed["ID"].astype(str)) & set(fig_grid["ID"].astype(str))
    print(
        f"figueira same-file overlap {len(fig_overlap)}/20 "
        f"routed[:5]={fig_routed['ID'].head(5).tolist()} grid[:5]={fig_grid['ID'].head(5).tolist()}"
    )

    section("B-grok-02 slug vs id (parallel success check)")
    from logic.src.pipeline.simulations.day_context import get_full_policy_name, to_slug
    from logic.src.utils.infrastructure.setup_sims import get_pol_name

    cases = [
        ("lookahead_na_amgat_emp", {"mandatory_selection": "lookahead", "route_improvement": [], "acceptance_criteria": None}),
        ("lookahead_alns_fast_tsp_emp", {"mandatory_selection": {"other/ms_lookahead.yaml": "cf70"}, "route_improvement": "fast_tsp"}),
        ("last_minute_cf70_hgs_emp", {"mandatory_selection": "cf70", "route_improvement": None}),
    ]
    for pol_id, cfg in cases:
        display = get_full_policy_name(pol_id, cfg)
        slug = to_slug(display)
        worker_key = get_pol_name(pol_id)
        res = {slug: [1.0], "success": True}
        accepted = worker_key in res
        print(f"id={worker_key}\n  display={display}\n  slug={slug}\n  worker accepts result={accepted}")
        if accepted:
            print("  UNEXPECTED match")
            failures += 1

    section("B-grok-04 overflow recount")
    from logic.src.pipeline.simulations.bins.base import Bins

    bins = Bins.__new__(Bins)
    bins.n = 1
    bins.real_c = np.array([100.0])
    bins.c = np.array([100.0])
    bins.means = np.zeros(1)
    bins.std = np.zeros(1)
    bins.day_count = 0
    bins.square_diff = np.zeros(1)
    bins.lost = np.zeros(1)
    bins.inoverflow = np.zeros(1)
    bins.volume = 1.0
    bins.density = 100.0  # 1% == 1 kg
    bins.history = []
    bins.level_history = []
    bins.noise_variance = 0.0
    n_ov, _, _, lost = bins._process_filling(np.array([0.0]))
    print(f"already-full bin, zero new waste -> overflows={n_ov} lost_kg={lost} (lost should be 0)")
    if n_ov != 1 or lost != 0.0:
        print("UNEXPECTED overflow accounting")
        failures += 1
    n_ov2, _, _, lost2 = bins._process_filling(np.array([0.0]))
    print(f"second idle day -> overflows={n_ov2} cumulative_inoverflow={float(bins.inoverflow.sum())}")

    section("B-grok-03 / B-grok-05 time base")
    # finishing.py writes perf_counter()-tic; running.py sets tic = perf_counter()+run_time
    import time

    run_time = 12.5
    tic = time.perf_counter() + run_time
    reported = time.perf_counter() - tic
    print(f"resume formula reported time={reported:.4f} (previous elapsed was +{run_time})")
    if reported > 0:
        print("UNEXPECTED positive resume time")
        failures += 1

    section("filesystem filenames")
    missing = [
        "bins_waste/Rio_Maior_Sensores_2021_2024_cleaned_104.csv",
        "coordinates/coordinates104.csv",
        "bins_waste/old_out_crude_rate[both].csv",
        "coordinates/old_out_info[both].csv",
        "Coordinates.xlsx",
    ]
    present = [
        "bins_waste/old_out_crude_rate[riomaior].csv",
        "coordinates/old_out_info[riomaior].csv",
        "bins_waste/out_rate_crude[riomaior].csv",
        "coordinates/out_info[riomaior].csv",
        "bins_waste/out_rate_crude[figdafoz].csv",
        "coordinates/out_info[figdafoz].csv",
        "coordinates/Facilities.csv",
        "bins_waste/StockAndAccumulationRate.xlsx",
        "coordinates/Coordinates.xlsx",
    ]
    for rel in missing:
        exists = os.path.exists(os.path.join(DATA, rel))
        print(f"missing-expected {rel}: exists={exists}")
        if exists:
            failures += 1
    for rel in present:
        exists = os.path.exists(os.path.join(DATA, rel))
        print(f"present-expected {rel}: exists={exists}")
        if not exists:
            failures += 1

    section("B-grok-06 CTOP load uses percent")
    # collection.py adds bins.c[node-1] (percent) into a kg capacity check
    print("bins.c is documented as observed fill percent; CTOP cur_load adds it directly")

    print(f"\nDONE failures={failures}")
    return failures


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
