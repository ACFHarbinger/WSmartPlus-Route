"""
Action for routing policy execution.
"""

import random
import time
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .base import SimulationAction, _flatten_config


class PolicyExecutionAction(SimulationAction):
    """
    Executes the routing policy to generate collection tours.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the selected routing policy."""
        from logic.src.policies import PolicyFactory
        from logic.src.policies.base import PolicyRegistry

        # Ensure all adapters are imported so the registry is fully populated
        PolicyFactory.ensure_registered()

        # 1. GET FULL POLICY NAME (for identification)
        full_policy = str(context.get("full_policy") or "")

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = context.get("config", {})

        # 2b. INJECT SIMULATION SEED INTO POLICY CONFIG
        # CRITICAL: Use policy-specific seed for RNG isolation
        # If policy_seed is available (set by run_day), use it; otherwise fallback to base seed
        policy_seed = context.get("policy_seed")
        if policy_seed is not None:
            seed = policy_seed
        else:
            # Fallback to base simulation seed
            sim_cfg = getattr(context.get("cfg"), "sim", None)
            seed = getattr(sim_cfg, "seed", 42) if sim_cfg else 42

        # Convert to a standard dict to ensure mutability
        if isinstance(raw_cfg, DictConfig):
            raw_cfg = OmegaConf.to_container(raw_cfg, resolve=True)
            context["config"] = raw_cfg

        raw_cfg["seed"] = seed

        # RE-SEED GLOBAL STATE BEFORE EXECUTION
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

            start_time = time.process_time()
            tour, cost, extra_output = adapter.execute(**context)
            elapsed_time = time.process_time() - start_time

            context["tour"] = tour
            context["cost"] = cost
            context["extra_output"] = extra_output
            context["time"] = elapsed_time

            # Legacy caching for regular selection
            if "regular" in full_policy:
                context["cached"] = extra_output

        except ValueError as e:
            raise ValueError(f"Failed to load policy adapter for '{solver_key}': {e}") from e
