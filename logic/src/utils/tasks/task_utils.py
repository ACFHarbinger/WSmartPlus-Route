"""
Utilities for problem instance creation and graph edge calculation.

This module provides functions to processing spatial coordinates,
generating edges based on distance or KNN strategies, and formatting
problem instances for the model.

Attributes:
    calculate_edges: Calculate edges for a problem instance.
    make_instance_generic: Generic instance creator.

Example:
    >>> import task_utils
    >>> task_utils.make_instance_generic(args, edge_threshold, edge_strategy)
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from logic.src.utils.graph import (
    adj_to_idx,
    get_adj_knn,
    get_edge_idx_dist,
)


def calculate_edges(
    loc: Union[np.ndarray, torch.Tensor], edge_threshold: float, edge_strategy: str
) -> Optional[torch.Tensor]:
    """
    Utility to calculate edges for a problem instance.

    Args:
        loc (Union[np.ndarray, torch.Tensor]): Location coordinates.
        edge_threshold (float): Edge threshold.
        edge_strategy (str): Edge strategy.

    Returns:
        torch.Tensor: Edge indices.
    """
    if edge_threshold <= 0:
        return None

    distance_matrix = squareform(pdist(loc, metric="euclidean"))
    if edge_strategy == "dist":
        return torch.tensor(get_edge_idx_dist(distance_matrix, edge_threshold)).to(dtype=torch.long)
    elif edge_strategy == "knn":
        neg_adj_matrix = get_adj_knn(distance_matrix, edge_threshold)
        return torch.tensor(adj_to_idx(neg_adj_matrix)).to(dtype=torch.long)
    return None


def make_instance_generic(args, edge_threshold: float, edge_strategy: str) -> Dict[str, torch.Tensor]:
    """
    Generic instance creator.

    Args:
        args (tuple): Arguments for the instance creator.
        edge_threshold (float): Edge threshold.
        edge_strategy (str): Edge strategy.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the instance.
    """
    depot, loc, waste, max_waste, *rest = args
    ret_dict = {
        "loc": torch.FloatTensor(loc),
        "depot": torch.FloatTensor(depot),
        "waste": torch.tensor(waste, dtype=torch.float),
        "max_waste": torch.tensor(max_waste, dtype=torch.float),
    }

    # Handle multi-day fill levels if present
    if ret_dict["waste"].size(dim=0) > 1:
        for day_id in range(1, len(waste)):
            ret_dict[f"fill{day_id}"] = ret_dict["waste"][day_id]
        ret_dict["waste"] = ret_dict["waste"][0]
    elif len(ret_dict["waste"].size()) > 1:
        ret_dict["waste"] = ret_dict["waste"][0]

    edges = calculate_edges(ret_dict["loc"], edge_threshold, edge_strategy)
    if edges is not None:
        ret_dict["edges"] = edges

    return ret_dict
