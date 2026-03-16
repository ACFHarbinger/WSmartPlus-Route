"""
Policy Summary Callback.

Display a summary of the policies that will be run in the simulation.
"""

import sys
from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.table import Table

from logic.src.configs import Config, MustGoConfig
from logic.src.interfaces import ITraversable
from logic.src.pipeline.simulations.actions.base import _flatten_config
from logic.src.tracking.logging.logger_writer import LoggerWriter


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
        table.add_column("Post-Processing", style="blue")

        policies: List[str] = cfg.sim.full_policies or []
        config_paths: Dict[str, Any] = dict(cfg.sim.config_path) if cfg.sim.config_path else {}

        for idx, policy_name in enumerate(policies):
            # Resolve config
            config = config_paths.get(policy_name, {})
            if isinstance(config, str):
                from logic.src.utils.configs.config_loader import load_config

                try:
                    config = load_config(config)
                except Exception:
                    config = {}

            # Extract details
            engine = self._extract_engine(policy_name, config)
            selection = self._extract_selection(config)
            post_proc = self._extract_post_processing(config)

            table.add_row(str(idx + 1), policy_name, engine, selection, post_proc)

        console.print(table)
        console.print("\n")

    def _extract_engine(self, policy_name: str, config: Dict[str, Any]) -> str:
        """Extract the engine name."""
        raw_cfg = config
        solver_key = None
        known_policy_keys = ["vrpp", "cvrp", "tsp", "hgs", "alns", "bpc", "sans", "neural"]

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
        config_must_go = flat_cfg.get("must_go")

        strategies = []

        items = []
        if isinstance(config_must_go, MustGoConfig):
            items = [config_must_go]
        elif isinstance(config_must_go, (list, tuple)):
            items = list(config_must_go)
        elif not isinstance(config_must_go, (str, dict, type(None))) and hasattr(config_must_go, "__iter__"):
            # config_must_go is Iterable but not str/dict/None
            from typing import Iterable, cast

            items = list(cast(Iterable[Any], config_must_go))
        elif config_must_go is not None:
            items = [config_must_go]

        for item in items:
            name, params = self._parse_selection_item(item)
            strategies.append(f"{name}{params}")

        return ", ".join(strategies) if strategies else "None"

    def _parse_selection_item(self, item: Any) -> tuple[str, str]:
        """Parse a single selection item into name and params."""
        name: str = "Unknown"
        params: str = ""

        item_obj: object = item
        if isinstance(item_obj, MustGoConfig):
            name = str(item_obj.strategy)  # type: ignore[assignment]
            if item_obj.strategy == "regular":
                params = f"(freq={item_obj.frequency})"
            elif item_obj.strategy == "revenue":
                params = f"(thresh={item_obj.revenue_threshold})"
        elif isinstance(item_obj, ITraversable):
            if "strategy" in item_obj:
                name = str(item_obj.get("strategy") or "Unknown")
                if "threshold" in item_obj:
                    params = f"(t={item_obj['threshold']})"
            else:
                item_keys = list(item_obj.keys())
                if item_keys:
                    name = str(item_keys[0])
                    val = item_obj[name]
                    val_obj: object = val
                    if isinstance(val_obj, ITraversable) and "threshold" in val_obj:
                        params = f"(t={val_obj['threshold']})"
        elif isinstance(item_obj, str):
            name = item_obj

        return name, params

    def _extract_post_processing(self, config: Dict[str, Any]) -> str:
        """Extract post processing steps."""
        raw_cfg = config
        flat_cfg = _flatten_config(raw_cfg)
        pp = flat_cfg.get("post_processing")

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
