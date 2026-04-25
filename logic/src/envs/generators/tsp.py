"""
Traveling Salesman Problem (TSP) instance generator.

Attributes:
    TSPGenerator: TSPGenerator class.

Example:
    >>> from logic.src.envs.generators import TSPGenerator
    >>> generator = TSPGenerator(num_loc=20)
    >>> instance = generator.generate()
    >>> instance
    TensorDict(
        {
            'locs': TensorDict(
                # ... customer locations (shape: [*B, 20, 2]) ...
            ),
            'depot': TensorDict(
                # ... depot location (shape: [*B, 2]) ...
            )
        }
    )
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .base import Generator


class TSPGenerator(Generator):
    """
    Generator for Traveling Salesman Problem (TSP) instances.

    Simple generator for TSP benchmarking with just locations.

    Attributes:
        num_loc: Number of customer locations.
    """

    def __init__(self, num_loc: int = 20, **kwargs):
        """Initialize TSPGenerator.

        Args:
            num_loc: Number of customer locations.
            kwargs: Forwarded to Generator base class.
        """
        super().__init__(num_loc=num_loc, **kwargs)

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate TSP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with:
                locs: [*B, num_loc, 2]     — customer locations
                depot: [*B, 2]              — depot location
        """
        # Generate locations (including depot as first location)
        all_locs: torch.Tensor = self._generate_locations(batch_size, self.num_loc + 1)

        return TensorDict(
            {
                "locs": all_locs.narrow(dim=-2, start=1, length=self.num_loc),  # Customer locations
                "depot": all_locs.select(dim=-2, index=0),  # First location as depot
            },
            batch_size=batch_size,
            device=self.device,
        )
