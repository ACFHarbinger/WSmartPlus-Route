"""
Action for must-go bin selection.
"""

import os
from typing import Any, Dict, List, cast

import numpy as np

from logic.src.configs import MustGoConfig
from logic.src.interfaces import ITraversable
from logic.src.constants import ROOT_DIR
from logic.src.interfaces import IBinContainer
from logic.src.policies.other import MustGoSelectionFactory, SelectionContext
from logic.src.utils.configs.config_loader import load_config

from .base import SimulationAction, _flatten_config


class MustGoSelectionAction(SimulationAction):
    """
    Identifies which bins are targets for collection based on various strategies.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Identifies bins that MUST be collected on the current day.
        """
        # 1. Gather all strategies to run
        strategies: List[Dict[str, Any]] = []

        # Check config for 'must_go' list
        raw_cfg = context.get("config", {})
        flat_cfg = _flatten_config(raw_cfg)
        config_must_go = flat_cfg.get("must_go")

        if isinstance(config_must_go, MustGoConfig):
            # Use the structured config object directly
            strategies.append({"config": config_must_go})
        elif config_must_go:
            if not isinstance(config_must_go, list):
                config_must_go = [config_must_go]

            for item in config_must_go:
                if isinstance(item, MustGoConfig):
                    strategies.append({"config": item})
                elif isinstance(item, str) and (item.endswith(".xml") or item.endswith(".yaml")):
                    # Load config file
                    fpath = os.path.join(ROOT_DIR, "assets", "configs", "policies", item)
                    cfg = load_config(fpath)
                    # Extract strategy name and params
                    if "config" in cfg and len(cfg) == 1:
                        cfg = cfg["config"]

                    if "strategy" in cfg:
                        strat_name = cfg.pop("strategy")
                        strategies.append({"name": strat_name, "params": cfg})
                    else:
                        for k, v in cfg.items():
                            strategies.append({"name": k, "params": v if isinstance(v, ITraversable) else {}})

                elif isinstance(item, ITraversable):
                    # Handle case where item is a dict describing a single strategy
                    if "strategy" in item:
                        strat_name = item.pop("strategy")
                        strategies.append({"name": strat_name, "params": item})
                    else:
                        for k, v in item.items():
                            strategies.append({"name": k, "params": v if isinstance(v, ITraversable) else {}})
                else:
                    strategies.append({"name": item, "params": {}})

        # 2. Execute all strategies and union results
        bins = context["bins"]
        # Use IBinContainer protocol for unified bin access
        if isinstance(bins, IBinContainer):
            current_fill = bins.get("c")
            accumulation_rates = bins.get("means")
            std_deviations = bins.get("std")
        else:
            # Fallback for dict-like objects
            current_fill = bins.get("c") if hasattr(bins, "get") else getattr(bins, "c", None)
            accumulation_rates = bins.get("means") if hasattr(bins, "get") else getattr(bins, "means", None)
            std_deviations = bins.get("std") if hasattr(bins, "get") else getattr(bins, "std", None)

        n_bins = getattr(bins, "n", None)
        if n_bins is None:
            n_bins = len(current_fill) if current_fill is not None else 0

        final_must_go = set()

        for strat_info in strategies:
            s_name = strat_info["name"]
            s_params = strat_info["params"]

            thresh = 0.0
            if isinstance(s_params, ITraversable):
                val = s_params.get("threshold") or s_params.get("cf") or s_params.get("param") or s_params.get("lvl")
                thresh = float(val) if val is not None else 0.0

            sel_ctx = SelectionContext(
                bin_ids=np.arange(0, n_bins, dtype="int32"),
                current_fill=np.array(current_fill) if current_fill is not None else np.array([]),
                accumulation_rates=accumulation_rates,
                std_deviations=std_deviations,
                current_day=context.get("day", 0),
                threshold=float(thresh) if thresh is not None else 0.0,
                next_collection_day=context.get("next_collection_day"),
                distance_matrix=context.get("distance_matrix"),
                paths_between_states=context.get("paths_between_states"),
                vehicle_capacity=context.get("max_capacity", 100.0),
            )

            if "config" in strat_info:
                # Use structured config
                m_config = strat_info["config"]
                strategy = MustGoSelectionFactory.create_from_config(m_config)
                res = strategy.select_bins(sel_ctx)
                s_name = m_config.strategy
            elif s_name == "select_all":
                res = list(sel_ctx.bin_ids)
            else:
                strategy = MustGoSelectionFactory.create_strategy(str(s_name), **cast(Dict[str, Any], s_params))
                res = strategy.select_bins(sel_ctx)

            # Ensure list
            if hasattr(res, "tolist"):
                res = res.tolist()
            elif not isinstance(res, list):
                res = list(res)

            final_must_go.update(res)

        context["must_go"] = list(final_must_go)
