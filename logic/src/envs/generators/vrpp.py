"""
VRPP problem generator.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict

from .base import Generator


class VRPPGenerator(Generator):
    """
    Generator for Vehicle Routing Problem with Profits (VRPP) instances.

    Generates instances with:
    - Depot location
    - Customer locations
    - Waste at each location
    - Waste values for visiting each location
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
        capacity: float = 1.0,
        max_length: Optional[float] = None,
        depot_type: str = "corner",
        device: Union[str, torch.device] = "cpu",
        area: Optional[str] = None,
        data_dir: Optional[str] = None,
        indices: Optional[list[int]] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize VRPP generator.

        Args:
            num_loc: Number of customer locations.
            min_loc: Minimum coordinate value.
            max_loc: Maximum coordinate value.
            loc_distribution: Distribution for location generation.
            min_waste: Minimum waste value.
            max_waste: Maximum waste value.
            waste_distribution: Distribution for waste generation.
            capacity: Vehicle capacity.
            max_length: Maximum route length (None for unlimited).
            depot_type: Depot placement ("center", "corner", "random").
            device: Device to place tensors on.
            area: Area for grid generation.
            data_dir: Directory for grid data.
            indices: Indices for grid generation.
            generator: Generator for random number generation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, generator, **kwargs)

        self.min_waste = min_waste
        self.max_waste = max_waste
        self.waste_distribution = kwargs.get("data_distribution", waste_distribution)
        self.capacity = capacity if capacity is not None else 1.0
        self.max_length = max_length
        self.depot_type = depot_type
        self.data_dir = data_dir
        self.area = area
        self.indices = indices if indices is not None else list(range(0, num_loc))
        try:
            from logic.src.utils.data.loader import load_grid_base

            self.grid = load_grid_base(self.indices, self.area, self.data_dir)
        except FileNotFoundError:
            self.grid = None

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate VRPP instances."""
        # Generate locations
        locs = self._generate_locations(batch_size)

        # Generate depot
        depot = self._generate_depot(batch_size)

        # Generate waste
        waste = self._generate_waste(batch_size)

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
        """Generate waste values."""
        # Use common utility for consistency
        from logic.src.data.generators.waste import generate_waste

        bs = math.prod(batch_size) if batch_size else 1
        coords = (
            self._generate_depot(batch_size).view(bs, 2),
            self._generate_locations(batch_size).view(bs, self.num_loc, 2),
        )
        waste = generate_waste(
            self.num_loc,
            self.waste_distribution,
            coords,
            bs,
            grid=self.grid,
            rng=self.rng,
            sample_method="sample_array",
        )
        if isinstance(waste, np.ndarray):
            waste = torch.from_numpy(waste).float()
        return waste.to(self.device).view(*batch_size, self.num_loc)
