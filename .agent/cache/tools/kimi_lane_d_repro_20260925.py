"""Lane D (Kimi) reproducer — minimal-export review, commit 70e660b03.

Run from a worktree checkout of 70e660b03 with data/ available:
    /home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python \
        .agent/cache/tools/kimi_lane_d_repro_20260925.py

Checks (all evidence-only, no source edits):
  1. B-kimi-01  NeuralAgentPolicy.execute returns profit in wrong units
                (fill-percent x EUR/kg) and cost computed on distC*100;
                both are discarded by the only retained caller
                (RouteConstructionAction unpacking `tour, _, _, ...`).
  2. B-kimi-02  `vrpp` default mismatch: RouteConstructionAction defaults
                vrpp to False (short-circuits empty mandatory to [0,0]) while
                BaseRoutingPolicy.execute defaults use_all_bins True (VRPP
                all-bins subset) -- latent because the action runs first.
  3. R-kimi evidence: BaseMultiPeriodRoutingPolicy has zero subclasses;
                compute_batch_sim has zero callers; _log_solver_params /
                policy_na._log_params bodies are unreachable (`run = None`).
  4. scenario-tree dead work: no retained policy consumes context["scenario_tree"].
"""

import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# 1. NA profit/cost unit inconsistency (live code, discarded result)
# --------------------------------------------------------------------------
import numpy as np
import torch

from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params
from logic.src.policies.route_construction.learning_algorithms.neural_agent import policy_na
from logic.src.policies.route_construction.learning_algorithms.neural_agent import simulation as na_sim

Q, R, B, C, V = load_area_and_waste_type_params("Rio Maior", "plastic")

bins = SimpleNamespace(
    c=np.array([50.0, 80.0]),  # fill PERCENT
    get_level_history=lambda device=None: torch.zeros(1, 2, 2),
)

policy = policy_na.NeuralAgentPolicy(config={"na": [{"decoding": {"strategy": "greedy"}}]})

# Capture real source BEFORE stubbing the agent's day computation.
sim_src = inspect.getsource(na_sim.SimulationMixin.compute_simulator_day)

# Stub out the agent's day computation: tour visits bins 1 and 2, cost 5.0.
na_sim.SimulationMixin.compute_simulator_day = lambda self, *a, **k: ([0, 1, 2, 0], 5.0, {})

kwargs = dict(
    model_env=SimpleNamespace(problem=None),
    model_ls=({"waste": torch.zeros(1, 2)}, (None, None), {"revenue_kg": R}),
    bins=bins,
    device=torch.device("cpu"),
    fill=np.array([50.0, 80.0]),
    dm_tensor=torch.ones(3, 3),
    config={"na": [{"decoding": {"strategy": "greedy"}}]},
    mandatory=[1, 2],
)

tour, cost, profit, _, _ = policy_na.NeuralAgentPolicy.execute(policy, **kwargs)

collected_pct = 50.0 + 80.0
profit_as_written = collected_pct * R - 5.0 * 1.0  # policy_na.py:136-139
profit_kg_basis = (collected_pct / 100.0) * V * B * R - 5.0 * 1.0  # c/100*V*B = kg

print(f"  R={R}, B={B}, V={V}: profit_as_written={profit_as_written:.4f} "
      f"profit_kg_basis={profit_kg_basis:.4f} ratio={profit_as_written / profit_kg_basis:.1f}x")
check("na-profit-matches-written-formula", abs(profit - profit_as_written) < 1e-6)
check("na-profit-unit-mismatch", abs(profit - profit_kg_basis) > 1e-6,
      "returned profit uses fill-percent x EUR/kg, not kg")

# Both values are discarded by the only retained caller:
import logic.src.pipeline.simulations.actions.route_construction as rc
src = inspect.getsource(rc.RouteConstructionAction.execute)
check("caller-discards-cost-profit", "tour, _, _, extra_output" in src,
      "RouteConstructionAction keeps only tour + extra_output")

# cost = get_route_cost(distC * 100, route)  (simulation.py:247)
check("na-cost-times-100", "distC * 100" in sim_src, "cost inflated 100x, then discarded")

# --------------------------------------------------------------------------
# 2. vrpp default mismatch
# --------------------------------------------------------------------------
check("action-vrpp-default-false", 'flat_cfg.get("vrpp", False)' in src)
base_src = inspect.getsource(
    __import__(
        "logic.src.policies.route_construction.base.base_routing_policy",
        fromlist=["BaseRoutingPolicy"],
    ).BaseRoutingPolicy.execute
)
check("base-vrpp-default-true", 'values.get("vrpp", True)' in base_src)

# --------------------------------------------------------------------------
# 3. Removal evidence: zero-subclass / zero-caller / unreachable bodies
# --------------------------------------------------------------------------
import subprocess

def rg(pattern, paths):
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py", pattern] + paths,
        capture_output=True, text=True,
    ).stdout
    return [l for l in out.splitlines() if l]

logic = "logic"
hits = rg("BaseMultiPeriodRoutingPolicy", [logic, "main.py"])
importers = [h for h in hits if "base_multi_period_policy" not in h and "base/__init__" not in h]
check("multi-period-base-zero-subclass", len(importers) == 0,
      f"unexpected importers: {importers}")

hits = rg("compute_batch_sim", [logic, "main.py"])
callers = [h for h in hits if "neural_agent/batch.py" not in h]
check("compute_batch_sim-zero-callers", len(callers) == 0, f"unexpected callers: {callers}")

brp_src = inspect.getsource(
    __import__(
        "logic.src.policies.route_construction.base.base_routing_policy",
        fromlist=["BaseRoutingPolicy"],
    ).BaseRoutingPolicy._log_solver_params
)
check("log-solver-params-unreachable", "run = None" in brp_src)
pna_src = inspect.getsource(policy_na.NeuralAgentPolicy._log_params)
check("na-log-params-unreachable", "run = None" in pna_src)

# --------------------------------------------------------------------------
# 4. scenario_tree has no consumer among retained policies
# --------------------------------------------------------------------------
hits = rg("scenario_tree", [logic])
consumers = [
    h for h in hits
    if "base_multi_period_policy" not in h
    and "prediction.py" not in h
    and "actions/route_construction.py" not in h
    and "joint_context" not in h
    and "selection_context" not in h
    and "interfaces/context" not in h  # ProblemContext.from_kwargs chain is only invoked by the dead base class
    and "recreate_repair" not in h  # optional params, never passed a tree
]
check("scenario-tree-no-retained-consumer", len(consumers) == 0,
      f"unexpected consumers: {consumers}")

print()
if FAILURES:
    print("FAILURES:", FAILURES)
    sys.exit(1)
print("all lane-D checks passed")
