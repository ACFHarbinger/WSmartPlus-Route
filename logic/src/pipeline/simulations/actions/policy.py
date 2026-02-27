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
        from logic.src.policies.adapters.registry import PolicyRegistry

        # 1. GET FULL POLICY NAME (for identification)
        full_policy = str(context.get("full_policy") or "")

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = context.get("config", {})
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

        # If the key is purely a selection strategy, default to 'tsp'
        registered = set(PolicyRegistry.list_policies())
        if solver_key and solver_key not in registered:
            from logic.src.policies.other.must_go.base.selection_registry import MustGoSelectionRegistry

            selection_keys = set(MustGoSelectionRegistry.list_strategies())
            # Check if it's a selection-only name
            lower = str(solver_key).lower()
            if any(sel in lower for sel in selection_keys) and not any(
                f"_{eng}" in lower or f"{eng}_" in lower or eng == lower for eng in registered
            ):
                solver_key = "tsp"
            else:
                # Try to extract a registered engine from the compound name
                for eng in sorted(registered, key=len, reverse=True):
                    if f"_{eng}" in lower or f"{eng}_" in lower or eng == lower:
                        solver_key = eng
                        break

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
