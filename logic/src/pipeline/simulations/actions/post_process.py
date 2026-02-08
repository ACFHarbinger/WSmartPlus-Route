"""
Action for tour post-processing and refinement.
"""

import os
from typing import Any, Dict

from loguru import logger

from logic.src.configs import PostProcessingConfig
from logic.src.constants import ROOT_DIR
from logic.src.utils.configs.config_loader import load_config

from .base import SimulationAction, _flatten_config


class PostProcessAction(SimulationAction):
    """
    Refines generated collection tours using modular refinement strategies.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Refine the generated collection tour."""
        tour = context.get("tour")
        if not tour or len(tour) <= 2:
            return

        raw_cfg = context.get("config", {})
        flat_cfg = _flatten_config(raw_cfg)

        pp_list = flat_cfg.get("post_processing") or context.get("post_process")

        if isinstance(pp_list, PostProcessingConfig):
            # Use structured config
            pp_configs = [pp_list]
        elif pp_list:
            if not isinstance(pp_list, list):
                if isinstance(pp_list, str) and pp_list.lower() != "none":
                    pp_list = [pp_list]
                else:
                    pp_list = []

            pp_configs = []
            for item in pp_list:
                if isinstance(item, PostProcessingConfig):
                    pp_configs.append(item)
                else:
                    pp_configs.append(item)
        else:
            pp_configs = []

        if pp_configs:
            from logic.src.policies.other.post_processing import PostProcessorFactory

            for entry in pp_configs:
                if isinstance(entry, PostProcessingConfig):
                    processors = PostProcessorFactory.create_from_config(entry)
                else:
                    # Legacy handling for strings/files
                    item = entry
                    pp_name = ""
                    pp_params = {k: v for k, v in context.items() if k != "tour"}

                    if isinstance(item, str) and (item.endswith(".xml") or item.endswith(".yaml")):
                        fpath = os.path.join(ROOT_DIR, "assets", "configs", "policies", item)
                        try:
                            cfg = load_config(fpath)
                            if "config" in cfg and len(cfg) == 1:
                                cfg = cfg["config"]

                            for k, v in cfg.items():
                                pp_name = k
                                if isinstance(v, dict):
                                    pp_params.update(v)
                        except (OSError, ValueError) as e:
                            logger.warning(f"Error loading post_processing config {item}: {e}")
                            continue
                    else:
                        pp_name = item

                    if not pp_name or pp_name.lower() == "none":
                        continue

                    try:
                        processors = [PostProcessorFactory.create(pp_name)]
                    except Exception as e:
                        logger.warning(f"Failed to create post-processor {pp_name}: {e}")
                        continue

                for processor in processors:
                    try:
                        # For structured config, we might need to pass params from context too
                        # or expect the factory to have handled it.
                        pf_params = {k: v for k, v in context.items() if k != "tour"}
                        refined_tour = processor.process(tour, **pf_params)

                        if refined_tour != tour:
                            from logic.src.policies.tsp import get_route_cost

                            dist_matrix = context.get("distance_matrix")
                            new_cost = get_route_cost(dist_matrix, refined_tour)

                            tour = refined_tour
                            context["tour"] = refined_tour
                            context["cost"] = new_cost
                    except Exception as e:
                        logger.warning(f"Post-processing skipped due to error: {e}")
