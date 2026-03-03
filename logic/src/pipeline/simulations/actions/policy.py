"""
Action for routing policy execution.
"""

import random
from typing import Any, Dict

import numpy as np
import torch

from logic.src.policies import PolicyFactory

from .base import SimulationAction, _flatten_config


class PolicyExecutionAction(SimulationAction):
    """
    Executes the routing policy to generate collection tours.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the selected routing policy."""
        from logic.src.policies.adapters.registry import PolicyRegistry

        # Ensure all adapters are imported so the registry is fully populated
        PolicyFactory.ensure_registered()

        # 1. GET FULL POLICY NAME (for identification)
        full_policy = str(context.get("full_policy") or "")

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = context.get("config", {})

        # 2b. INJECT SIMULATION SEED INTO POLICY CONFIG
        # This ensures that both global and explicit seeding are available
        sim_cfg = getattr(context.get("cfg"), "sim", None)
        base_seed = getattr(sim_cfg, "seed", 42) if sim_cfg else 42
        sample_id = context.get("sample_id", 0)
        day = context.get("day", 0)

        # Unique but deterministic seed for this (sample, day) pair
        # Formula ensures no overlap for typical simulation scales
        seed = base_seed + (day * 1000) + (sample_id * 1000000)
        raw_cfg["seed"] = seed

        # RE-SEED GLOBAL STATE BEFORE EXECUTION
        # This guarantees that even if a worker was "dirtied" by a previous run,
        # each day's execution starts from a deterministic state.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        flat_cfg = _flatten_config(raw_cfg)

        # Explicit engine/solver from config
        solver_key = flat_cfg.get("policy.type") or flat_cfg.get("policy.solver") or flat_cfg.get("policy.engine")

        # Fallback: identify solver from keys in raw config matching registry
        if not solver_key:
            registered = set(PolicyRegistry.list_policies())
            for key in registered:
                if key in raw_cfg:
                    solver_key = key
                    break

        # Last resort: use full_policy string
        if not solver_key:
            solver_key = full_policy

        # Resolve compound names (e.g. 'lookahead_lahc_custom_emp' -> 'lahc')
        registered = set(PolicyRegistry.list_policies())
        if solver_key and solver_key not in registered:
            lower = str(solver_key).lower()
            # Try to extract a registered engine from the compound name
            resolved = False
            for eng in sorted(registered, key=len, reverse=True):
                if f"_{eng}" in lower or f"{eng}_" in lower or eng == lower:
                    solver_key = eng
                    resolved = True
                    break
            if not resolved:
                raise ValueError(
                    f"Unknown policy '{solver_key}' (from '{full_policy}'). Registered policies: {sorted(registered)}"
                )

        # If no nodes to collect, skip policy and return to depot
        must_go = context.get("must_go", [])
        if not must_go:
            context["tour"] = [0, 0]
            context["cost"] = 0.0
            context["extra_output"] = None
            return

        try:
            adapter = PolicyFactory.get_adapter(solver_key, config=raw_cfg)
            tour, cost, extra_output = adapter.execute(**context)

            context["tour"] = tour
            context["cost"] = cost
            context["extra_output"] = extra_output

            # Legacy caching for regular selection
            if "regular" in full_policy:
                context["cached"] = extra_output

        except ValueError as e:
            raise ValueError(f"Failed to load policy adapter for '{solver_key}': {e}") from e
