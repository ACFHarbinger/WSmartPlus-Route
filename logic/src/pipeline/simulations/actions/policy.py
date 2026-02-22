"""
Action for routing policy execution.
"""

from typing import Any, Dict

from logic.src.policies import PolicyFactory

from .base import SimulationAction, _flatten_config


class PolicyExecutionAction(SimulationAction):
    """
    Executes the routing policy to generate collection tours.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the selected routing policy."""
        # 1. GET FULL POLICY NAME (for identification)
        full_policy = str(context.get("full_policy") or "")

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = context.get("config", {})
        flat_cfg = _flatten_config(raw_cfg)

        # Explicit engine/solver from config
        solver_key = flat_cfg.get("policy.type") or flat_cfg.get("policy.solver") or flat_cfg.get("policy.engine")

        # Fallback to identify solver from keys in raw config (standard Hydra behavior)
        if not solver_key:
            known_policy_keys = [
                "vrpp",
                "cvrp",
                "tsp",
                "hgs_alns",
                "hgs",
                "alns",
                "bcp",
                "sans",
                "ils",
                "lkh",
                "aco",
                "sisr",
                "neural",
            ]
            for key in known_policy_keys:
                if key in raw_cfg:
                    solver_key = key
                    break

        # Last resort fallback to full_policy string (if it's a simple name like 'hgs')
        if not solver_key:
            solver_key = full_policy

        # Standardize 'tsp' for selection strategies that don't specify a solver
        selection_keys = ["regular", "last_minute", "select_all", "lookahead", "means_std", "revenue", "service_level"]
        engine_keys = [
            "hgs",
            "alns",
            "hgs_alns",
            "vrpp",
            "neural",
            "bcp",
            "sans",
            "ils",
            "lkh",
            "aco",
            "hh_aco",
            "ks_aco",
            "sisr",
            "cvrp",
            "tsp",
            "am",
            "ptr",
            "ddam",
            "ahvpl",
            "hvpl",
        ]

        # Prioritize exact matches in Registry to avoid false positives (e.g. 'gamma' contains 'am')
        from logic.src.policies.adapters.registry import PolicyRegistry

        # If solver_key is already a valid registered policy, keep it
        if solver_key and (PolicyRegistry.get(str(solver_key)) or PolicyRegistry.get(f"policy_{solver_key}")):
            pass
        elif any(f"{k}" in str(solver_key).lower() for k in selection_keys) and not any(
            # Robust check: engine must be preceded/followed by _ or be a whole word
            f"_{k}" in str(solver_key).lower() or f"{k}_" in str(solver_key).lower() or k == str(solver_key).lower()
            for k in engine_keys
        ):
            solver_key = "tsp"
        # Final robust fallback: if still unknown or it's an expanded name, try to find an engine keyword
        elif solver_key not in engine_keys:
            for engine in engine_keys:
                # Robust match: engine word or part of _snake_case_
                if (
                    f"_{engine}" in str(solver_key).lower()
                    or f"{engine}_" in str(solver_key).lower()
                    or engine == str(solver_key).lower()
                ):
                    # Handle special mapping for neural
                    solver_key = "neural" if engine in ["am", "ptr", "ddam"] else engine
                    break

        # If no nodes to collect, skip policy and return to depot
        must_go = context.get("must_go", [])
        if not must_go:
            context["tour"] = [0, 0]
            context["cost"] = 0.0
            context["extra_output"] = None
            return

        try:
            # We no longer pass 'engine' and 'threshold' as separate args from here
            # because they should be inside 'context' or 'config' handled by adapter
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
