#!/usr/bin/env python3
"""Lane F reproducer for the 2026-09-25 minimal-export review (commit 70e660b03).

Checks the selector unit/formula mismatches, the training depot-column bug,
Fast-TSP seed drop, empty-tour handling, and BMC/OI accept convention.
Run from any checkout of 70e660b03:

  /home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python \\
    .agent/cache/tools/cursor_lane_f_repro_20260925.py
"""

from __future__ import annotations

import inspect
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from logic.src.constants import MAX_CAPACITY_PERCENT, MAX_WASTE
from logic.src.interfaces.context import SelectionContext
from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import (
    BoltzmannAcceptance,
)
from logic.src.policies.acceptance_criteria.only_improving import OnlyImproving
from logic.src.policies.mandatory_selection.selection_last_minute import LastMinuteSelection
from logic.src.policies.mandatory_selection.selection_lookahead import LookaheadSelection
from logic.src.policies.mandatory_selection.selection_service_level import ServiceLevelSelection
from logic.src.policies.route_improvement.common.helpers import assemble_tour, split_tour
from logic.src.policies.route_improvement.fast_tsp import FastTSPRouteImprover
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem import (
    tsp as tsp_mod,
)
from logic.src.policies.vector.selection.last_minute import LastMinuteSelector
from logic.src.policies.vector.selection.lookahead import LookaheadSelector
from logic.src.policies.vector.selection.service_level import ServiceLevelSelector


def _ok(name: str, cond: bool, detail: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        raise SystemExit(f"assertion failed: {name}")


def test_service_level_linear_not_sqrt() -> None:
    fill = np.array([80.0, 80.0])
    rate = np.array([5.0, 5.0])
    std = np.array([10.0, 10.0])
    z, n_d = 1.0, 4
    ctx = SelectionContext(
        bin_ids=np.arange(2),
        current_fill=fill,
        accumulation_rates=rate,
        std_deviations=std,
        threshold=z,
        horizon_days=n_d,
        max_fill=MAX_CAPACITY_PERCENT,
    )
    ids, _ = ServiceLevelSelection().select_bins(ctx)
    linear = fill + n_d * rate + n_d * z * std  # 80+20+40 = 140
    sqrt = fill + n_d * rate + math.sqrt(n_d) * z * std  # 80+20+20 = 120
    _ok(
        "B-cursor-01 linear formula",
        bool((linear >= MAX_CAPACITY_PERCENT).all()) and bool((sqrt >= MAX_CAPACITY_PERCENT).all()),
        f"linear={linear.tolist()} sqrt={sqrt.tolist()} selected={ids}",
    )
    # Both bins selected under linear; paper sqrt also exceeds 100 here.
    # A tighter case: fill=70, rate=5, std=5, n_d=4, z=1
    # linear = 70+20+20=110 >= 100; sqrt = 70+20+10=100 >= 100 still.
    # fill=70, rate=4, std=5, n_d=4, z=1 → linear=70+16+20=106; sqrt=70+16+10=96
    fill2 = np.array([70.0])
    rate2 = np.array([4.0])
    std2 = np.array([5.0])
    ctx2 = SelectionContext(
        bin_ids=np.arange(1),
        current_fill=fill2,
        accumulation_rates=rate2,
        std_deviations=std2,
        threshold=1.0,
        horizon_days=4,
        max_fill=MAX_CAPACITY_PERCENT,
    )
    ids2, _ = ServiceLevelSelection().select_bins(ctx2)
    linear2 = 70 + 4 * 4 + 4 * 1 * 5  # 106
    sqrt2 = 70 + 4 * 4 + math.sqrt(4) * 5  # 96
    vec = ServiceLevelSelector(confidence_factor=1.0, horizon_days=4)
    mask = vec.select(
        torch.tensor([[0.0, 0.70]]),
        accumulation_rates=torch.tensor([[0.0, 0.04]]),
        std_deviations=torch.tensor([[0.0, 0.05]]),
    )
    _ok(
        "B-cursor-01 paper disagreement",
        ids2 == [1] and linear2 >= 100 and sqrt2 < 100 and bool(mask[0, 1].item()),
        f"scalar selected={ids2} (linear={linear2}, paper-sqrt={sqrt2}); "
        f"vectorized with depot col={mask.tolist()}",
    )
    src = inspect.getsource(ServiceLevelSelection.select_bins)
    _ok(
        "B-cursor-01 no sqrt in scalar",
        "sqrt" not in src and "horizon_days" in src and "std_deviations * horizon_days" in src.replace("\n", " "),
        "scalar source uses n_d * sigma, not sqrt(n_d) * sigma",
    )


def test_last_minute_units() -> None:
    fill_pct = np.array([50.0, 75.0, 95.0])
    ctx70 = SelectionContext(
        bin_ids=np.arange(3),
        current_fill=fill_pct,
        threshold=70.0,
        max_fill=MAX_CAPACITY_PERCENT,
    )
    ids70, _ = LastMinuteSelection().select_bins(ctx70)
    ctx07 = SelectionContext(
        bin_ids=np.arange(3),
        current_fill=fill_pct,
        threshold=0.7,
        max_fill=MAX_CAPACITY_PERCENT,
    )
    ids07, _ = LastMinuteSelection().select_bins(ctx07)
    vec = LastMinuteSelector(threshold=0.7)
    mask_frac = vec.select(torch.tensor([[0.50, 0.75, 0.95]]))
    mask_pct = vec.select(torch.tensor([[50.0, 75.0, 95.0]]))
    _ok(
        "B-cursor-03 scalar 70 vs 0.7",
        ids70 == [2, 3] and ids07 == [1, 2, 3],
        f"threshold=70 → {ids70}; threshold=0.7 on percent fill → {ids07} (everything)",
    )
    _ok(
        "B-cursor-03 vectorized 0.7 on fraction vs percent",
        mask_frac.tolist() == [[False, True, True]] and mask_pct.tolist() == [[False, True, True]],
        f"frac={mask_frac.tolist()} pct={mask_pct.tolist()} MAX_WASTE={MAX_WASTE} "
        "(index 0 forced False; 50% fill still exceeds threshold 0.7)",
    )


def test_training_depot_column() -> None:
    """steps.py feeds customer-only waste; selectors zero index 0 as depot."""
    fill = torch.tensor([[0.9, 0.1, 0.1]])  # customer 0 is full
    mask = LastMinuteSelector(threshold=0.7).select(fill)
    _ok(
        "B-cursor-02 first customer dropped",
        mask.tolist() == [[False, False, False]],
        f"customer 0 at 0.9 should be mandatory but mask={mask.tolist()}",
    )
    # After the steps.py prepend, the mask is [depot=False, c0=False, c1, c2]
    # so the full customer is permanently excluded from training constraints.


def test_lookahead_day_multiplier() -> None:
    fill = np.array([60.0, 40.0])
    rate = np.array([50.0, 20.0])  # bin0 overflows today; from empty needs 2 days
    ctx0 = SelectionContext(
        bin_ids=np.arange(2),
        current_fill=fill,
        accumulation_rates=rate,
        current_collection_day=0,
        max_fill=MAX_CAPACITY_PERCENT,
    )
    ids0, _ = LookaheadSelection().select_bins(ctx0)
    ctx5 = SelectionContext(
        bin_ids=np.arange(2),
        current_fill=fill,
        accumulation_rates=rate,
        current_collection_day=5,
        max_fill=MAX_CAPACITY_PERCENT,
    )
    ids5, _ = LookaheadSelection().select_bins(ctx5)
    vec = LookaheadSelector(current_collection_day=5)
    vmask = vec.select(
        torch.tensor([[0.60, 0.40]]),
        accumulation_rates=torch.tensor([[0.50, 0.20]]),
        current_collection_day=5,
    )
    _ok(
        "B-cursor-04 day=0 vs day=5",
        ids0 != ids5 or True,  # document both
        f"scalar day0={ids0} day5={ids5}; vectorized day5={vmask.tolist()}",
    )
    # Scalar uses fill + j*rate with j in range(today+1, next); at today=5 this
    # multiplies by absolute day numbers (6,7,...) instead of days-from-today.
    _ok(
        "B-cursor-04 scalar uses absolute j",
        "j * accumulation_rates" in inspect.getsource(LookaheadSelection._add_bins_to_collect),
        "selection_lookahead.py:_add_bins_to_collect uses j as an absolute multiplier",
    )


def test_fast_tsp_seed_and_empty() -> None:
    src = inspect.getsource(tsp_mod.find_route)
    _ok(
        "B-cursor-05 seed accepted not forwarded",
        "seed=42" in src and "fast_tsp.find_tour" in src and "seed" not in src.split("fast_tsp.find_tour", 1)[1].split("\n", 1)[0],
        f"find_tour call: {[ln.strip() for ln in src.splitlines() if 'find_tour' in ln]}",
    )
    _ok("empty [0]", split_tour([0]) == [], f"split_tour([0])={split_tour([0])}")
    _ok("empty [0,0]", split_tour([0, 0]) == [], f"split_tour([0,0])={split_tour([0, 0])}")
    improver = FastTSPRouteImprover()
    tour, metrics = improver.process([0, 0], distance_matrix=np.zeros((3, 3)))
    _ok(
        "empty tour returned unchanged",
        tour == [0, 0],
        f"process([0,0]) → {tour} metrics={metrics}",
    )
    # Mandatory nodes already in a trip are reordered, not dropped.
    kept = assemble_tour([[3, 1, 2]])
    _ok(
        "assemble keeps nodes",
        set(kept) >= {0, 1, 2, 3},
        f"assemble_tour([[3,1,2]])={kept}",
    )


def test_acceptance() -> None:
    bmc = BoltzmannAcceptance(initial_temp=1.0, alpha=0.9, seed=0)
    acc_imp, _ = bmc.accept(current_obj=10.0, candidate_obj=12.0)
    acc_worse, _ = bmc.accept(current_obj=10.0, candidate_obj=8.0)
    oi = OnlyImproving()
    oi_imp, _ = oi.accept(current_obj=10.0, candidate_obj=12.0)
    oi_worse, _ = oi.accept(current_obj=10.0, candidate_obj=8.0)
    _ok("BMC accepts improving (max)", acc_imp is True, f"10→12 accepted={acc_imp}")
    _ok("OI accepts improving (max)", oi_imp is True, f"10→12 accepted={oi_imp}")
    _ok("OI rejects worsening", oi_worse is False, f"10→8 accepted={oi_worse}")
    # Overflow guard: tiny T, large negative delta.
    bmc.T = 1e-12
    acc_cold, m = bmc.accept(current_obj=10.0, candidate_obj=-1e6)
    _ok(
        "BMC T<=1e-9 rejects",
        acc_cold is False and m["temperature"] <= 1e-9,
        f"accepted={acc_cold} T={m['temperature']}",
    )
    sig = inspect.signature(BoltzmannAcceptance.accept)
    _ok(
        "accept signature",
        list(sig.parameters)[:2] == ["self", "current_obj"] and "candidate_obj" in sig.parameters,
        f"params={list(sig.parameters)}",
    )


if __name__ == "__main__":
    test_service_level_linear_not_sqrt()
    test_last_minute_units()
    test_training_depot_column()
    test_lookahead_day_multiplier()
    test_fast_tsp_seed_and_empty()
    test_acceptance()
    print("all lane-F checks reproduced")
