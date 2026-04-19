"""
Pickup and Delivery Problem (PDP) generator.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from tensordict import TensorDict

from .base import Generator


class PDPGenerator(Generator):
    """
    Generator for the Pickup and Delivery Problem (PDP).

    Produces instances with:
    - locs  : [*B, num_loc, 2]  — pickup and delivery coordinates
    - depot : [*B, 2]           — depot coordinate

    Node layout (after depot prepend in the environment):
      [0] = depot
      [1  …  num_loc/2]        = pickup nodes
      [num_loc/2+1  …  num_loc] = delivery nodes

    Each pickup node i has a paired delivery at i + num_loc/2.

    Note: ``num_loc`` must be even.  If odd, it is incremented by 1.
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        rng: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise PDPGenerator.

        Args:
            num_loc: Number of customer locations (excl. depot, must be even).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution.
            depot_type: Depot placement ("center", "corner", "random").
            device: Target device.
            rng: Optional numpy RNG.
            generator: Optional torch RNG.
            **kwargs: Forwarded to Generator base.
        """
        if num_loc % 2 != 0:
            num_loc += 1  # PDP requires even number of customer nodes
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            device=device,
            rng=rng,
            generator=generator,
            **kwargs,
        )
        self.depot_type = depot_type

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate a batch of PDP instances."""
        locs = self._generate_locations(batch_size)
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
        """Generate depot location."""
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2.0
            return torch.full((*batch_size, 2), center, device=self.device, dtype=torch.float32)
        elif self.depot_type == "corner":
            return torch.full((*batch_size, 2), self.min_loc, device=self.device, dtype=torch.float32)
        else:  # random
            return self._uniform_locations(batch_size, 1).squeeze(-2)
