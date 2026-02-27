"""
Traveling Salesman Problem (TSP) instance generator.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .base import Generator


class TSPGenerator(Generator):
    """
    Generator for Traveling Salesman Problem (TSP) instances.

    Simple generator for TSP benchmarking with just locations.
    """

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate TSP instances."""
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
