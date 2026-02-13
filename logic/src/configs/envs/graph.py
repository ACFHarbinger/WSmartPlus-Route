"""
Graph/instance configuration module.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphConfig:
    """Configuration for problem instances and graph data.

    Attributes:
        area: County area of the bins locations.
        waste_type: Type of waste bins selected for the optimization problem.
        vertex_method: Method to transform vertex coordinates ('mmn', etc.).
        distance_method: Method to compute distance matrix ('ogd', etc.).
        dm_filepath: Path to the distance matrix file.
        edge_threshold: How many of all possible edges to consider.
        edge_method: Method for getting edges ('dist', 'knn', etc.).
        focus_graph: Paths to the files with the coordinates of the graphs to focus on.
        focus_size: Number of focus graphs to include.
        eval_focus_size: Number of focus graphs to include in evaluation.
    """

    area: str = "riomaior"
    num_loc: int = 50
    waste_type: str = "plastic"
    vertex_method: str = "mmn"
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    edge_threshold: str = "0"
    edge_method: Optional[str] = None
    focus_graph: Optional[str] = None
    focus_size: Optional[int] = None
    eval_focus_size: Optional[int] = None
