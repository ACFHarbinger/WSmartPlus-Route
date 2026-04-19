"""
Orienteering Problem (OP) generator.

Prize types follow Fischetti et al. (1998) and Kool et al. (2019).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from tensordict import TensorDict

from .base import Generator

# Default max-length table (from Kool et al. 2019)
MAX_LENGTHS: dict[int, float] = {20: 2.0, 50: 3.0, 100: 4.0}


class OPGenerator(Generator):
    """
    Generator for the Orienteering Problem (OP).

    Produces instances with:
    - locs       : [*B, num_loc, 2]  — customer coordinates
    - depot      : [*B, 2]           — depot coordinate
    - prize      : [*B, num_loc]     — prize at each customer
    - max_length : [*B]              — maximum path length allowed
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        prize_type: str = "dist",
        max_length: Optional[float] = None,
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        rng: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise OPGenerator.

        Args:
            num_loc: Number of customer locations (excluding depot).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution.
            prize_type: Prize generation method:
                "dist" — proportional to distance from depot,
                "unif" — uniform in [0.01, 1.0],
                "const" — constant 1.0.
            max_length: Maximum allowed path length. Defaults to lookup table.
            depot_type: Depot placement ("center", "corner", "random").
            device: Target device.
            rng: Optional numpy RNG.
            generator: Optional torch RNG.
            **kwargs: Forwarded to Generator base.
        """
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
        self.prize_type = prize_type
        self.depot_type = depot_type

        if max_length is None:
            closest = min(MAX_LENGTHS.keys(), key=lambda x: abs(x - num_loc))
            max_length = MAX_LENGTHS.get(num_loc, MAX_LENGTHS[closest])
        self.max_length = max_length

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate a batch of OP instances."""
        locs = self._generate_locations(batch_size)
        depot = self._generate_depot(batch_size)
        locs_with_depot = torch.cat([depot.unsqueeze(-2), locs], dim=-2)

        if self.prize_type == "const":
            prize = torch.ones(*batch_size, self.num_loc, device=self.device, dtype=torch.float32)
        elif self.prize_type == "unif":
            prize = (
                1
                + torch.randint(
                    0, 100, (*batch_size, self.num_loc), device=self.device, generator=self.generator
                ).float()
            ) / 100.0
        else:  # "dist"
            dist_to_depot = (locs_with_depot[..., 0:1, :] - locs_with_depot[..., 1:, :]).norm(p=2, dim=-1)
            prize = (1 + (dist_to_depot / dist_to_depot.max(dim=-1, keepdim=True)[0] * 99).int().float()) / 100.0

        max_length = torch.full((*batch_size,), self.max_length, device=self.device, dtype=torch.float32)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "prize": prize,
                "max_length": max_length,
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
