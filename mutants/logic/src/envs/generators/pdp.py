"""
Pickup and Delivery Problem (PDP) instance generator.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import torch
from tensordict import TensorDict

from .base import Generator


class PDPGenerator(Generator):
    """
    Generator for Pickup and Delivery Problem (PDP) instances.

    Generates instances with paired pickup and delivery locations.
    The first N locations are pickups, and the next N locations are
    their corresponding deliveries (i.e., i and i+N are paired).
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        depot_type: str = "center",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize PDP generator.

        Args:
            num_loc: Number of pickup nodes (so total nodes = 2 * num_loc).
                     Note: This differs from VRP where num_loc is total customers.
            min_loc: Minimum coordinate value.
            max_loc: Maximum coordinate value.
            loc_distribution: Distribution for location generation.
            depot_type: Depot placement ("center", "corner", "random").
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, **kwargs)
        self.depot_type = depot_type

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate PDP instances."""
        # Generate 2 * num_loc locations (pickups + deliveries)
        # We explicitly generate 2 * N locations
        total_locs = 2 * self.num_loc
        locs = self._generate_locations(batch_size, total_locs)

        # Generate depot
        depot = self._generate_depot(batch_size)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
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
