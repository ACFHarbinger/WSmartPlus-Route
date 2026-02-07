"""
Action for tour post-processing and refinement.
"""

import os
from typing import Any, Dict

from logic.src.constants import ROOT_DIR
from logic.src.utils.configs.config_loader import load_config
from loguru import logger

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

        if pp_list:
            if not isinstance(pp_list, list):
                if isinstance(pp_list, str) and pp_list.lower() != "none":
                    pp_list = [pp_list]
                else:
                    pp_list = []

            from logic.src.policies.post_processing import PostProcessorFactory

            for item in pp_list:
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
                    processor = PostProcessorFactory.create(pp_name)
                    refined_tour = processor.process(tour, **pp_params)

                    if refined_tour != tour:
                        from logic.src.policies.single_vehicle import get_route_cost

                        dist_matrix = context.get("distance_matrix")
                        new_cost = get_route_cost(dist_matrix, refined_tour)

                        tour = refined_tour
                        context["tour"] = refined_tour
                        context["cost"] = new_cost

                except Exception as e:
                    logger.warning(f"Post-processing {pp_name} skipped due to error: {e}")
