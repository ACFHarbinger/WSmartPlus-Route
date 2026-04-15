"""
HGS Policy Adapter.

Adapts the Hybrid Genetic Search (HGS) logic to the common policy interface.
Now agnostic to bin selection.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HGSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .dispatcher import run_hgs


@PolicyRegistry.register("hgs")
class HGSPolicy(BaseRoutingPolicy):
    """
    Hybrid Genetic Search policy class.

    Visits pre-selected 'must_go' bins using evolutionary optimization.
    """

    def __init__(self, config: Optional[Union[HGSConfig, Dict[str, Any]]] = None):
        """Initialize HGS policy with optional config.

        Args:
            config: HGSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSConfig

    def _get_config_key(self) -> str:
        """Return config key for HGS."""
        return "hgs"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run HGS solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Project lat/lng to a local equirectangular plane centred on the depot (index 0).
        # This is required for correct polar sector angles in SWAP* pruning.
        # Raw lat/lng cannot be used directly in atan2 — longitude degrees are shorter
        # than latitude degrees by a factor of cos(lat), causing ~22% distortion at 39°N.
        x_coords_proj: Optional[np.ndarray] = None
        y_coords_proj: Optional[np.ndarray] = None

        if x_coords is not None and y_coords is not None:
            depot_lat = float(y_coords[0])
            depot_lng = float(x_coords[0])
            cos_lat = math.cos(math.radians(depot_lat))
            x_coords_proj = (x_coords - depot_lng) * cos_lat
            y_coords_proj = y_coords - depot_lat

        routes, profit, solver_cost = run_hgs(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes=mandatory_nodes,
            x_coords=x_coords_proj,
            y_coords=y_coords_proj,
        )
        return routes, profit, solver_cost
