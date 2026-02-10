"""
Action for tour post-processing and refinement.
"""

import os
from typing import Any, Dict

from loguru import logger

from logic.src.configs import PostProcessingConfig
from logic.src.constants import ROOT_DIR
from logic.src.interfaces import ITraversable
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

        pp_configs = self._get_post_processing_configs(context)

        if pp_configs:
            for entry in pp_configs:
                processors = self._create_processors(entry, context)
                self._apply_processors(processors, context)

    def _get_post_processing_configs(self, context: Dict[str, Any]) -> list:
        """Retrieve and normalize post-processing configurations."""
        raw_cfg = context.get("config", {})
        flat_cfg = _flatten_config(raw_cfg)
        pp_list = flat_cfg.get("post_processing") or context.get("post_process")

        if isinstance(pp_list, PostProcessingConfig):
            return [pp_list]

        if not pp_list:
            return []

        if not isinstance(pp_list, list):
            pp_list = [pp_list] if isinstance(pp_list, str) and pp_list.lower() != "none" else []

        pp_configs = []
        for item in pp_list:
            pp_configs.append(item)
        return pp_configs

    def _create_processors(self, entry: Any, context: Dict[str, Any]) -> list:
        """Create processors from a configuration entry."""
        from logic.src.policies.other.post_processing import PostProcessorFactory

        if isinstance(entry, PostProcessingConfig):
            return PostProcessorFactory.create_from_config(entry)

        # Legacy handling for strings/files
        return self._create_legacy_processors(entry, context, PostProcessorFactory)

    def _create_legacy_processors(self, item: Any, context: Dict[str, Any], factory) -> list:
        """Handle legacy string/file configuration for processors."""
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
                    if isinstance(v, ITraversable):
                        pp_params.update(v)
            except (OSError, ValueError) as e:
                logger.warning(f"Error loading post_processing config {item}: {e}")
                return []
        else:
            pp_name = item

        if not pp_name or (isinstance(pp_name, str) and pp_name.lower() == "none"):
            return []

        try:
            # We can't easily pass params here to create(), maybe factory handles it?
            # The original code only passed pp_name to create().
            return [factory.create(pp_name)]
        except Exception as e:
            logger.warning(f"Failed to create post-processor {pp_name}: {e}")
            return []

    def _apply_processors(self, processors: list, context: Dict[str, Any]) -> None:
        """Apply a list of processors to the tour in the context."""
        tour = context.get("tour")

        for processor in processors:
            try:
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
