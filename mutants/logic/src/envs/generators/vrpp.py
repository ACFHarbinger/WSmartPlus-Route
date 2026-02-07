"""
VRPP problem generator.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from logic.src.utils.data.data_utils import generate_waste_prize
from tensordict import TensorDict

from .base import Generator


class VRPPGenerator(Generator):
    """
    Generator for Vehicle Routing Problem with Profits (VRPP) instances.

    Generates instances with:
    - Depot location
    - Customer locations
    - Waste/demand at each location
    - Prizes (revenue) for visiting each location
    - Vehicle capacity
    - Maximum route length
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        min_waste: float = 0.0,
        max_waste: float = 1.0,
        waste_distribution: str = "uniform",
        min_prize: float = 0.0,
        max_prize: float = 1.0,
        prize_distribution: str = "uniform",
        capacity: float = 1.0,
        max_length: Optional[float] = None,
        depot_type: str = "center",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize VRPP generator.

        Args:
            num_loc: Number of customer locations.
            min_loc: Minimum coordinate value.
            max_loc: Maximum coordinate value.
            loc_distribution: Distribution for location generation.
            min_waste: Minimum waste/demand value.
            max_waste: Maximum waste/demand value.
            waste_distribution: Distribution for waste generation.
            min_prize: Minimum prize value.
            max_prize: Maximum prize value.
            prize_distribution: Distribution for prize generation.
            capacity: Vehicle capacity.
            max_length: Maximum route length (None for unlimited).
            depot_type: Depot placement ("center", "corner", "random").
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, **kwargs)

        self.min_waste = min_waste
        self.max_waste = max_waste
        self.waste_distribution = kwargs.get("data_distribution", waste_distribution)
        self.min_prize = min_prize
        self.max_prize = max_prize
        self.prize_distribution = kwargs.get("data_distribution", prize_distribution)
        self.capacity = capacity if capacity is not None else 1.0
        self.max_length = max_length
        self.depot_type = depot_type

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate VRPP instances."""
        # Generate locations
        locs = self._generate_locations(batch_size)

        # Generate depot
        depot = self._generate_depot(batch_size)

        # Generate waste/demand
        waste = self._generate_waste(batch_size)

        # Generate prizes
        prize = self._generate_prize(batch_size)

        # Compute max_waste per instance (for normalization)
        max_waste = torch.full((*batch_size,), self.capacity, device=self.device)

        # Compute max_length if not specified
        max_length = self.max_length
        if max_length is None:
            # Default based on problem size (heuristic)
            max_length = 2.0 + (self.num_loc / 50.0)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "waste": waste,
                "prize": prize,
                "capacity": torch.full((*batch_size,), self.capacity, device=self.device),
                "max_waste": max_waste,
                "max_length": torch.full((*batch_size,), max_length, device=self.device),
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

    def _generate_waste(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate waste/demand values."""
        # Use common utility for consistency
        bs = batch_size[0] if batch_size else 1
        coords = (self._generate_depot(batch_size), self._generate_locations(batch_size))
        waste = generate_waste_prize(self.num_loc, self.waste_distribution, coords, bs, bins=self.bins)
        if isinstance(waste, np.ndarray):
            waste = torch.from_numpy(waste).float()
        return waste.to(self.device).view(*batch_size, self.num_loc)

    def _generate_prize(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate prize values."""
        if self.prize_distribution == "distance_correlated":
            # Distance correlation still handled locally for now
            depot = self._generate_depot(batch_size)
            locs = self._generate_locations(batch_size)
            distances = torch.norm(locs - depot.unsqueeze(-2), dim=-1)
            max_dist = distances.max(dim=-1, keepdim=True).values
            normalized_dist = distances / (max_dist + 1e-8)
            return normalized_dist * (self.max_prize - self.min_prize) + self.min_prize

        # Otherwise use common utility
        bs = batch_size[0] if batch_size else 1
        coords = (self._generate_depot(batch_size), self._generate_locations(batch_size))
        prize = generate_waste_prize(self.num_loc, self.prize_distribution, coords, bs, bins=self.bins)
        if isinstance(prize, np.ndarray):
            prize = torch.from_numpy(prize).float()
        return prize.to(self.device).view(*batch_size, self.num_loc)
