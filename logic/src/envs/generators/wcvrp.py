"""
WCVRP problem generator.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Union

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.utils.data.data_utils import generate_waste

from .base import Generator


class WCVRPGenerator(Generator):
    """
    Generator for Waste Collection VRP (WCVRP) instances.

    Similar to VRPP but focused on waste collection with:
    - Bin fill levels that can change over time
    - Collection thresholds
    - Cost structure for collection
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        min_fill: float = 0.0,
        max_fill: float = 1.0,
        fill_distribution: str = "uniform",
        capacity: float = 100.0,
        cost_km: float = 1.0,
        revenue_kg: float = 0.1625,
        depot_type: str = "center",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize WCVRP generator.

        Args:
            num_loc: Number of bin locations.
            min_loc: Minimum coordinate value.
            max_loc: Maximum coordinate value.
            loc_distribution: Distribution for location generation.
            min_fill: Minimum fill level.
            max_fill: Maximum fill level.
            fill_distribution: Distribution for fill level generation.
            capacity: Vehicle capacity in kg.
            cost_km: Cost per kilometer traveled.
            revenue_kg: Revenue per kg collected.
            depot_type: Depot placement ("center", "corner", "random").
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, **kwargs)

        self.min_fill = min_fill
        self.max_fill = max_fill
        self.fill_distribution = kwargs.get("data_distribution", fill_distribution)
        self.capacity = capacity if capacity is not None else 100.0
        self.cost_km = cost_km
        self.revenue_kg = revenue_kg
        self.depot_type = depot_type

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate WCVRP instances."""
        # Generate locations
        locs = self._generate_locations(batch_size)

        # Generate depot
        depot = self._generate_depot(batch_size)

        # Generate fill levels (waste amount at each bin)
        fill = self._generate_fill_levels(batch_size)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "waste": fill,  # Standardized keyword
                "capacity": torch.full((*batch_size,), self.capacity, device=self.device),
                "max_waste": torch.full(
                    (*batch_size,), 1.0, device=self.device
                ),  # Consistent with definitions.MAX_WASTE
                "cost_km": torch.full((*batch_size,), self.cost_km, device=self.device),
                "revenue_kg": torch.full((*batch_size,), self.revenue_kg, device=self.device),
            },
            batch_size=batch_size,
            device=self.device,
        )

    def _generate_depot(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate depot location based on depot_type."""
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2
            return torch.full((*batch_size, 2), center, device=self.device)
        elif self.depot_type == "corner":
            return torch.full((*batch_size, 2), self.min_loc, device=self.device)
        elif self.depot_type == "random":
            return self._uniform_locations(batch_size, 1).squeeze(-2)
        else:
            raise ValueError(f"Unknown depot type: {self.depot_type}")

    def _generate_fill_levels(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate bin fill levels."""
        # Use common utility for consistency
        bs = math.prod(batch_size) if batch_size else 1
        coords = (
            self._generate_depot(batch_size).view(bs, 2),
            self._generate_locations(batch_size).view(bs, self.num_loc, 2),
        )
        fill = generate_waste(self.num_loc, self.fill_distribution, coords, bs, bins=self.bins)
        if isinstance(fill, np.ndarray):
            fill = torch.from_numpy(fill).float()
        return fill.to(self.device).view(*batch_size, self.num_loc)
