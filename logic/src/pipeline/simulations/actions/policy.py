"""
Action for routing policy execution.
"""

from typing import Any, Dict

from logic.src.interfaces import ITraversable
from logic.src.policies.adapters import PolicyFactory

from .base import SimulationAction, _flatten_config


class PolicyExecutionAction(SimulationAction):
    """
    Executes the routing policy to generate collection tours.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the selected routing policy."""
        policy_name = str(context.get("policy_name") or context.get("policy") or "")
        engine = context.get("engine")
        threshold = context.get("threshold")
        must_go = context.get("must_go", [])

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = context.get("config", {})

        solver_key = None
        known_policy_keys = ["vrpp", "cvrp", "tsp", "hgs_alns", "hgs", "alns", "bcp", "sans", "neural"]
        for key in known_policy_keys:
            if key in raw_cfg:
                solver_key = key
                break

        flat_cfg = _flatten_config(raw_cfg)
        policy_cfg = flat_cfg.get("policy", {})

        if not solver_key:
            if isinstance(policy_cfg, ITraversable):
                solver_key = policy_cfg.get("type") or policy_cfg.get("solver") or policy_cfg.get("engine")
            elif isinstance(policy_cfg, str):
                solver_key = policy_cfg

        if not solver_key and engine:
            solver_key = engine

        if not solver_key:
            solver_key = policy_name

        selection_keys = ["regular", "last_minute", "select_all", "lookahead", "means_std", "revenue"]
        if any(k in str(solver_key).lower() for k in selection_keys) and not any(
            k in str(solver_key).lower() for k in ["hgs", "alns", "vrpp", "neural", "bcp"]
        ):
            solver_key = "tsp"

        if (
            not must_go
            and solver_key != "neural"
            and not (str(solver_key).startswith("am") or str(solver_key).startswith("ddam"))
        ):
            context["tour"] = [0, 0]
            context["cost"] = 0.0
            context["extra_output"] = None
            return

        try:
            adapter = PolicyFactory.get_adapter(solver_key, engine=engine, threshold=threshold)
            tour, cost, extra_output = adapter.execute(**context)

            context["tour"] = tour
            context["cost"] = cost
            context["extra_output"] = extra_output

            if "regular" in policy_name:
                context["cached"] = extra_output
            elif solver_key == "neural" or str(solver_key).startswith("am"):
                context["output_dict"] = extra_output

        except ValueError as e:
            raise ValueError(f"Failed to load policy adapter for '{solver_key}': {e}") from e
