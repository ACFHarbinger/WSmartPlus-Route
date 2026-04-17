"""
Policy Summary Callback.

Display a summary of the policies that will be run in the simulation.
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

            table.add_row(str(idx + 1), policy_name, engine, selection, route_imp)

        console.print(table)
        console.print("\n")

    def _extract_engine(self, policy_name: str, config: Dict[str, Any]) -> str:
        """Extract the engine name."""
        raw_cfg = config
        solver_key = None
        known_policy_keys = ["vrpp", "cvrp", "tsp", "hgs", "alns", "bpc", "sans", "na"]

        # 1. Check top-level keys
        for key in known_policy_keys:
            if key in raw_cfg:
                solver_key = key
                break

        flat_cfg = _flatten_config(raw_cfg)
        policy_cfg = flat_cfg.get("policy", {})

        # 2. Check 'policy' sub-config
        if not solver_key:
            policy_cfg_obj: object = policy_cfg
            if isinstance(policy_cfg_obj, ITraversable):
                solver_key = policy_cfg_obj.get("type") or policy_cfg_obj.get("solver") or policy_cfg_obj.get("engine")
            elif isinstance(policy_cfg, str):
                solver_key = policy_cfg

        # 3. Fallback to name inference
        if not solver_key:
            for key in known_policy_keys:
                if key in policy_name:
                    solver_key = key
                    break

        return str(solver_key).upper() if solver_key else "Unknown"

    def _extract_selection(self, config: Dict[str, Any]) -> str:
        """Extract selection strategy details."""
        raw_cfg = config
        flat_cfg = _flatten_config(raw_cfg)
        config_mandatory = flat_cfg.get("mandatory")

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
            strategies.append(f"{name}{params}")

        return ", ".join(strategies) if strategies else "None"

    def _parse_selection_item(self, item: Any) -> tuple[str, str]:
        """Parse a single selection item into name and params."""
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

        return name, params

    def _parse_mandatory_config_params(self, config: MandatorySelectionConfig) -> str:
        """Extract formatted parameters from a MandatorySelectionConfig object."""
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
        """Extract formatted parameters from an ITraversable (Config/Dict) object."""
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
        """Extract route improvement steps."""
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
