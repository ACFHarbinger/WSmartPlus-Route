"""
Traveling Salesman Problem (TSP) instance generator.
"""

from __future__ import annotations

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
        all_locs = self._generate_locations(batch_size, self.num_loc + 1)

        return TensorDict(
            {
                "locs": all_locs[..., 1:, :],  # Customer locations
                "depot": all_locs[..., 0, :],  # First location as depot
            },
            batch_size=batch_size,
            device=self.device,
        )
