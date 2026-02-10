"""
Policy Summary Callback.

Display a summary of the policies that will be run in the simulation.
"""

from typing import Any, Dict

from rich import box
from rich.console import Console
from rich.table import Table

from logic.src.configs import MustGoConfig
from logic.src.interfaces import ITraversable
from logic.src.pipeline.simulations.actions.base import _flatten_config


class PolicySummaryCallback:
    """
    Callback to print a detailed summary of the policies.
    """

    def display(self, opts: Dict[str, Any]) -> None:
        """
        Display the policy summary table.

        Args:
            opts: The options dictionary containing 'policies' and 'config_path'.
        """
        console = Console()
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

        policies = opts.get("policies", [])
        config_paths = opts.get("config_path", {})

        for idx, policy_name in enumerate(policies):
            # Resolve config
            config = config_paths.get(policy_name, {})
            # If config is a path (str), we might need to load it, but orchestrator
            # seems to pass pre-loaded configs or paths. config.py expands them.
            # However, expand_policy_configs in config.py puts the actual config dict
            # or the path into opts["config_path"].
            # If it's a path, we'd strictly need to load it, but for summary we might start simple.
            # Actually, let's try to load it if it is a path to be robust.
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
        # Logic similar to PolicyExecutionAction
        raw_cfg = config
        solver_key = None
        known_policy_keys = ["vrpp", "cvrp", "tsp", "hgs", "alns", "bcp", "sans", "neural"]

        # 1. Check top-level keys
        for key in known_policy_keys:
            if key in raw_cfg:
                solver_key = key
                break

        flat_cfg = _flatten_config(raw_cfg)
        policy_cfg = flat_cfg.get("policy", {})

        # 2. Check 'policy' sub-config
        if not solver_key:
            if isinstance(policy_cfg, ITraversable):
                solver_key = policy_cfg.get("type") or policy_cfg.get("solver") or policy_cfg.get("engine")
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
        # Logic similar to MustGoSelectionAction
        raw_cfg = config
        flat_cfg = _flatten_config(raw_cfg)
        config_must_go = flat_cfg.get("must_go")

        strategies = []

        items = []
        if isinstance(config_must_go, MustGoConfig):
            items = [config_must_go]
        elif isinstance(config_must_go, list):
            items = config_must_go
        elif config_must_go:
            items = [config_must_go]

        for item in items:
            name = "Unknown"
            params = ""

            if isinstance(item, MustGoConfig):
                name = item.strategy
                if item.strategy == "lookahead":
                    pass  # No specific params essential for summary usually
                elif item.strategy == "regular":
                    params = f"(freq={item.frequency})"
                elif item.strategy == "revenue":
                    params = f"(thresh={item.revenue_threshold})"
            elif isinstance(item, ITraversable):
                if "strategy" in item:
                    name = item.get("strategy")
                    # Try to get threshold or relevant param
                    if "threshold" in item:
                        params = f"(t={item['threshold']})"
                else:
                    # Fallback if no strategy key, use keys as names (old behavior)
                    keys = list(item.keys())
                    if keys:
                        name = keys[0]
                        val = item[name]
                        if isinstance(val, ITraversable) and "threshold" in val:
                            params = f"(t={val['threshold']})"
            elif isinstance(item, str):
                name = item

            strategies.append(f"{name}{params}")

        return ", ".join(strategies) if strategies else "None"

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
            if isinstance(item, str):
                steps.append(item)
            elif isinstance(item, ITraversable):
                steps.append(list(item.keys())[0])

        return ", ".join(steps)
