"""Witnesses for Cursor P4 (Mandatory Selection) paper-update rows, 2026-09-27.

No simulator run. Reimplements the three shipped predicates in percent units
and checks archived yaml / paper tokens. Run from anywhere:

    python .agent/cache/tools/cursor_lane_f_ms_20260927.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path("/home/pkhunter/Repositories/Doc/WSmart-Route")
MAX = 100.0


def last_minute(fill: np.ndarray, threshold: float) -> np.ndarray:
    return fill >= threshold


def service_level(fill: np.ndarray, mu: np.ndarray, sigma: np.ndarray, z: float, n: int) -> np.ndarray:
    return fill + n * mu + z * n * sigma >= MAX


def lookahead_seed_and_bundle(fill: np.ndarray, rate: np.ndarray, today: int = 0) -> np.ndarray:
    n = len(fill)
    seed = fill + rate >= MAX
    mandatory = set(np.nonzero(seed)[0].tolist())
    if not mandatory:
        return seed
    tmp = fill.copy()
    for i in mandatory:
        tmp[i] = 0.0
    next_days = []
    for i in mandatory:
        if rate[i] <= 0:
            continue
        level = 0.0
        day = today
        while level < MAX:
            level += rate[i]
            day += 1
        next_days.append(day)
    if not next_days:
        return seed
    t_star = min(next_days)
    extra = np.zeros(n, dtype=bool)
    if t_star > today:
        for i in range(n):
            if i in mandatory:
                continue
            for j in range(today + 1, t_star):
                if fill[i] + (j - today) * rate[i] >= MAX:
                    extra[i] = True
                    break
    out = seed | extra
    return out


def check_lm_units() -> None:
    fill = np.array([50.0, 70.0, 90.0])
    assert last_minute(fill, 70.0).tolist() == [False, True, True]
    assert last_minute(fill, 90.0).tolist() == [False, False, True]
    # paper CF=0.7 is the ratio equivalent of threshold 70
    assert (fill / 100.0 >= 0.7).tolist() == last_minute(fill, 70.0).tolist()
    print("PASS  LM percent 70/90 == ratio 0.70/0.90")


def check_sl_linear() -> None:
    fill = np.array([60.0, 40.0])
    mu = np.array([20.0, 10.0])
    sigma = np.array([5.0, 8.0])
    sl1 = service_level(fill, mu, sigma, 0.84, 1)
    sl2 = service_level(fill, mu, sigma, 0.84, 2)
    # bin 0: 60+20+0.84*5 = 84.2 < 100; 60+40+1.68*5 = 108.4 >= 100
    assert sl1.tolist() == [False, False]
    assert sl2.tolist() == [True, False]
    # z=0, n=1 is the LA seed
    assert service_level(fill, mu, np.zeros(2), 0.0, 1).tolist() == (fill + mu >= MAX).tolist()
    print("PASS  SL linear n*z*sigma; SL2 doubles the buffer; z=0 n=1 == LA seed")


def check_la_bundle() -> None:
    fill = np.array([80.0, 40.0, 95.0])
    rate = np.array([25.0, 20.0, 10.0])
    mask = lookahead_seed_and_bundle(fill, rate, today=0)
    assert mask.tolist() == [True, True, True], mask
    quiet = lookahead_seed_and_bundle(np.array([10.0, 20.0]), np.array([5.0, 8.0]))
    assert quiet.tolist() == [False, False]
    print("PASS  LA seed {0,2} plus bundle {1}; quiet day is empty")


def check_files() -> None:
    eoq = (ROOT / "logic/src/policies/mandatory_selection/base/eoq.py").read_text()
    assert "fill_ratios is normalized" in eoq
    assert "return current_fill >= tau" in eoq
    lm = (ROOT / "logic/configs/policies/other/ms_last_minute.yaml").read_text()
    assert "threshold: 70" in lm and "threshold: 90" in lm
    sl = (ROOT / "logic/configs/policies/other/ms_service_level.yaml").read_text()
    assert "confidence_factor: 0.84" in sl
    assert "horizon_days: 1" in sl and "horizon_days: 2" in sl
    la = (ROOT / "logic/configs/policies/other/ms_lookahead.yaml").read_text()
    assert "current_collection_day: 0" in la
    assert "GRF" in la  # comment only; code has no GRF
    la_py = (ROOT / "logic/src/policies/mandatory_selection/selection_lookahead.py").read_text()
    assert "GRF" not in la_py
    archived = ROOT / "assets/output/30days/riomaior100_plastic/gamma3/lm_cls/hydra/pruned_config.yaml"
    text = archived.read_text()
    assert "noise_variance: 0.0" in text
    assert "stats_filepath: null" in text
    assert "psi: 1" in text
    paper = (
        ROOT
        / "assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection/paper.tex"
    ).read_text()
    assert "n_d" in paper
    assert r"\text{CF} = 0.7" in paper
    print("PASS  file tokens (yaml, eoq >=, archive psi=1, paper still has n_d)")


if __name__ == "__main__":
    check_lm_units()
    check_sl_linear()
    check_la_bundle()
    check_files()
    print("OK  cursor P4 witnesses")
