"""
Policy Summary Callback.

Display a summary of the policies that will be run in the simulation.

Attributes:
    PolicySummaryCallback: Callback to display a summary of the policies.

Example:
    >>> from logic.src.configs import Config
    >>> from logic.src.pipeline.callbacks.simulation.policy_summary import PolicySummaryCallback
    >>> callback = PolicySummaryCallback()
    >>> callback.display(cfg)
"""

import sys
from typing import Any, Dict, Iterable, List, cast

from rich import box
from rich.console import Console
from rich.table import Table

from logic.src.configs import Config, MandatorySelectionConfig
from logic.src.interfaces import ITraversable
from logic.src.pipeline.simulations.actions.base import _flatten_config
from logic.src.tracking.logging.logger_writer import LoggerWriter
from logic.src.utils.configs.config_loader import load_config


class PolicySummaryCallback:
    """
    Callback to print a detailed summary of the policies.

    Attributes:
        None
    """

    def display(self, cfg: Config) -> None:
        """
        Display the policy summary table.

        Args:
            cfg: Root Config with ``cfg.sim.full_policies`` and
                ``cfg.sim.config_path``.
        """
        main_console_out = sys.stdout
        if isinstance(main_console_out, LoggerWriter):
            main_console_out = main_console_out.terminal

        console = Console(file=main_console_out)
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title="Simulation Policies Summary",
            box=box.HEAVY_EDGE,
            padding=(0, 1),
            expand=False,
        )

        table.add_column("Idx", justify="right", style="dim")
        table.add_column("Policy Name", style="bold cyan")
        table.add_column("Engine", style="green")
        table.add_column("Selection Strategy", style="yellow")
        table.add_column("Route Improvement", style="blue")
        table.add_column("Acceptance", style="red")

        policies: List[str] = cfg.sim.full_policies or []
        config_paths: Dict[str, Any] = dict(cfg.sim.config_path) if cfg.sim.config_path else {}

        for idx, policy_name in enumerate(policies):
            # Resolve config
            config = config_paths.get(policy_name, {})
            if isinstance(config, str):
                try:
                    config = load_config(config)
                except Exception:
                    config = {}

            # Extract details
            engine = self._extract_engine(policy_name, config)
            selection = self._extract_selection(config)
            route_imp = self._extract_route_improvement(config)
            acceptance = self._extract_acceptance_criterion(config)

            table.add_row(str(idx + 1), policy_name, engine, selection, route_imp, acceptance)

        console.print(table)
        console.print("\n")

    def _extract_engine(self, policy_name: str, config: Dict[str, Any]) -> str:
        """Extract the engine name dynamically.

        Args:
            policy_name: Name of the policy.
            config: Configuration of the policy.

        Returns:
            The engine name.
        """
        raw_cfg = config
        solver_key = None

        # 1. Try to find engine in 'policy' sub-config
        flat_cfg = _flatten_config(raw_cfg)
        policy_cfg = flat_cfg.get("policy", {})
        if isinstance(policy_cfg, ITraversable):
            solver_key = policy_cfg.get("type") or policy_cfg.get("solver") or policy_cfg.get("engine")
        elif isinstance(policy_cfg, str):
            solver_key = policy_cfg

        # 2. Check top-level keys, skipping known non-engine keys
        if not solver_key:
            non_engine_keys = {"mandatory_selection", "route_improvement", "acceptance_criterion", "p", "setup"}
            for key in raw_cfg.keys():
                if key not in non_engine_keys:
                    solver_key = key
                    break

        # 3. Fallback to name inference from policy_name
        if not solver_key:
            solver_key = policy_name.split("_")[0]

        return str(solver_key).replace("_", "-").upper() if solver_key else "Unknown"

    def _extract_selection(self, config: Dict[str, Any]) -> str:
        """Extract selection strategy details.

        Args:
            config: Configuration of the policy.

        Returns:
            The selection strategy.
        """
        raw_cfg = config
        flat_cfg = _flatten_config(raw_cfg)
        config_mandatory = flat_cfg.get("mandatory_selection")

        strategies = []

        items = []
        if isinstance(config_mandatory, MandatorySelectionConfig):
            items = [config_mandatory]
        elif isinstance(config_mandatory, (list, tuple)):
            items = list(config_mandatory)
        elif not isinstance(config_mandatory, (str, dict, type(None))) and hasattr(config_mandatory, "__iter__"):
            # config_mandatory is Iterable but not str/dict/None
            items = list(cast(Iterable[Any], config_mandatory))
        elif config_mandatory is not None:
            items = [config_mandatory]

        for item in items:
            name, params = self._parse_selection_item(item)
            strategies.append("{}{}".format(name.title(), ":" + params if params else ""))

        return ", ".join(strategies) if strategies else "None"

    def _parse_selection_item(self, item: Any) -> tuple[str, str]:
        """Parse a single selection item into name and params.

        Args:
            item: Description of item.

        Returns:
            Description of return value.
        """
        name: str = "Unknown"
        params: str = ""

        item_obj: object = item
        if isinstance(item_obj, MandatorySelectionConfig):
            name = str(item_obj.strategy)
            params = self._parse_mandatory_config_params(item_obj)

        elif isinstance(item_obj, ITraversable):
            if "strategy" in item_obj:
                name = str(item_obj.get("strategy") or "Unknown")
                params = self._parse_traversable_params(item_obj)
            else:
                item_keys = list(item_obj.keys())
                if item_keys:
                    name = str(item_keys[0])
                    val = item_obj[name]
                    if isinstance(val, ITraversable):
                        params = self._parse_traversable_params(val)

        elif isinstance(item_obj, str):
            name = item_obj
            # Strip 'other/ms_', 'other/ri_' and '.yaml'
            if name.startswith("other/ms_"):
                name = name[len("other/ms_") :]
            if name.startswith("other/ri_"):
                name = name[len("other/ri_") :]
            if name.endswith(".yaml"):
                name = name[: -len(".yaml")]

        return name, params

    def _parse_mandatory_config_params(self, config: MandatorySelectionConfig) -> str:
        """Extract formatted parameters from a MandatorySelectionConfig object.

        Args:
            config: Configuration of the policy.

        Returns:
            Formatted parameters for the selection strategy.
        """
        strategy = config.strategy
        if strategy == "regular":
            return f"(freq={config.regular.frequency})"
        if strategy == "revenue":
            return f"(thresh={config.revenue.revenue_threshold})"
        if strategy == "service_level":
            return f"(cf={config.service_level.confidence_factor})"
        if strategy == "deadline":
            return f"(horizon={config.deadline.horizon_days}, t={config.deadline.threshold})"
        if strategy == "multi_day_prob":
            return f"(horizon={config.multi_day_prob.horizon_days}, t={config.multi_day_prob.threshold})"
        if strategy == "last_minute":
            return f"(t={config.last_minute.threshold})"
        if strategy == "lookahead":
            return f"(t={config.lookahead.threshold})"
        if strategy == "pareto_front":
            return f"(t={config.pareto_front.threshold})"
        if strategy == "profit_per_km":
            return f"(t={config.profit_per_km.threshold})"
        if strategy == "stochastic_regret":
            return f"(t={config.stochastic_regret.threshold})"
        if strategy == "spatial_synergy":
            s = config.spatial_synergy
            return f"(crit={s.critical_threshold}, syn={s.synergy_threshold}, r={s.radius})"
        if strategy == "mip_knapsack":
            return f"(v={config.mip_knapsack.n_vehicles}, p={config.mip_knapsack.overflow_penalty_frac})"
        return ""

    def _parse_traversable_params(self, item: ITraversable) -> str:
        """Extract formatted parameters from an ITraversable (Config/Dict) object.

        Args:
            item: Configuration of the policy.

        Returns:
            Formatted parameters for the selection strategy.
        """
        if "threshold" in item:
            return f"(t={item['threshold']})"
        if "horizon_days" in item:
            return f"(h={item['horizon_days']})"
        if "frequency" in item:
            return f"(f={item['frequency']})"
        if "critical_threshold" in item:
            return f"(c={item['critical_threshold']}, s={item.get('synergy_threshold')})"
        return ""

    def _extract_route_improvement(self, config: Dict[str, Any]) -> str:
        """Extract route improvement steps.

        Args:
            config: Configuration of the policy.

        Returns:
            Route improvement steps.
        """
        raw_cfg = config
        flat_cfg = _flatten_config(raw_cfg)
        pp = flat_cfg.get("route_improvement")

        if not pp:
            return "None"

        steps = []
        items = pp if isinstance(pp, list) else [pp]
        for item in items:
            item_obj: object = item
            if isinstance(item_obj, str):
                steps.append(item_obj)
            elif isinstance(item_obj, ITraversable):
                steps.append(list(item_obj.keys())[0])

        return ", ".join(steps)

    def _extract_acceptance_criterion(self, config: Dict[str, Any]) -> str:
        """Extract the acceptance criterion method.

        Args:
            config: Configuration of the policy.

        Returns:
            The acceptance criterion method.
        """
        raw_cfg = config
        flat_cfg = _flatten_config(raw_cfg)
        ac = flat_cfg.get("acceptance_criterion")

        if not ac:
            return "None"

        if isinstance(ac, dict):
            return str(ac.get("method", "Unknown")).upper()
        if isinstance(ac, ITraversable):
            return str(ac.get("method") or "Unknown").upper()
        return str(ac).upper()
