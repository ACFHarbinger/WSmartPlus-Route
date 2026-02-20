"""validators.py module.

Validation and sanitization for data generation configuration.
"""

import re
from typing import List, Optional

from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.constants import MAP_DEPOTS, WASTE_TYPES


def _sanitize_area(area: Optional[str]) -> str:
    if area is None:
        return "riomaior"
    area = re.sub(r"[^a-zA-Z]", "", area.lower())
    assert area in MAP_DEPOTS, f"Unknown area {area}, available areas: {list(MAP_DEPOTS.keys())}"
    return area


def _sanitize_waste(waste: Optional[str]) -> str:
    if waste is None:
        return "plastic"
    waste = re.sub(r"[^a-zA-Z]", "", waste.lower())
    assert waste in WASTE_TYPES, f"Unknown waste type {waste}, available types: {list(WASTE_TYPES.keys())}"
    return waste


def validate_data_config(cfg: Config) -> None:
    """
    Validates and sanitizes data generation configuration values in-place.

    Performs the same checks previously done by ``validate_gen_data_args``
    on the flattened opts dict, now applied directly to the typed Config.

    Args:
        cfg: Root Hydra configuration with ``cfg.data`` containing data
            generation parameters.

    Raises:
        AssertionError: If any validation constraint is violated.
    """
    data = cfg.data
    graphs: List[GraphConfig] = list(data.graphs) if data.graphs else []
    dataset_count = len(graphs)

    # --- Filename constraints ---
    if data.filename is not None:
        is_single_problem = isinstance(data.problem, str) and data.problem != "all"
        assert is_single_problem and dataset_count <= 1, "Can only specify filename when generating a single dataset"

    # --- Problem-specific validation ---
    if data.problem in ["all", "swcvrp"]:
        assert data.mu is not None, "Must specify mu when generating swcvrp datasets"
        assert data.sigma is not None, "Must specify sigma when generating swcvrp datasets"
        if isinstance(data.mu, list) and isinstance(data.sigma, list):
            assert len(data.mu) == len(data.sigma), "mu and sigma must have same length"

    # --- Sanitize graph configs ---
    for graph in graphs:
        graph.area = _sanitize_area(graph.area)
        graph.waste_type = _sanitize_waste(graph.waste_type)

        if graph.num_loc is None:
            graph.num_loc = 50
        if graph.vertex_method is None:
            graph.vertex_method = "mmn"
        if graph.focus_size is None:
            graph.focus_size = 31

    data.graphs = graphs


# Keep backward-compatible alias
validate_gen_data_args = validate_data_config
