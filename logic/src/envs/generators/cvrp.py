"""
CVRP problem generator.

Generates Capacitated Vehicle Routing Problem instances following the
benchmark conventions of Kool et al. (2019).

Attributes:
    CAPACITIES: dict[int, float]: Vehicle capacity table from Kool et al. (2019), Hottung et al. (2022), Kim et al. (2023)
    CVRPGenerator: Generator for the Capacitated Vehicle Routing Problem (CVRP).
        Produces instances with:
        - locs             : [*B, num_loc, 2]  — customer coordinates
        - depot            : [*B, 2]           — depot coordinate
        - demand           : [*B, num_loc]     — normalised demand in (0, 1]
        - vehicle_capacity : [*B]              — vehicle capacity (normalised to 1.0)

    Example:
        >>> from src.envs.generators import get_generator
        >>> generator = get_generator("cvrp", num_loc=20)
        >>> problem = generator.generate()
        >>> problem
        <ProblemInstance: ...>
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from tensordict import TensorDict

from .base import Generator

# Vehicle capacity table from Kool et al. (2019), Hottung et al. (2022), Kim et al. (2023)
CAPACITIES: dict[int, float] = {
    10: 20.0,
    15: 25.0,
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}


class CVRPGenerator(Generator):
    """
    Generator for the Capacitated Vehicle Routing Problem (CVRP).

    Produces instances with:
    - locs             : [*B, num_loc, 2]  — customer coordinates
    - depot            : [*B, 2]           — depot coordinate
    - demand           : [*B, num_loc]     — normalised demand in (0, 1]
    - vehicle_capacity : [*B]              — vehicle capacity (normalised to 1.0)

    Attributes:
        min_demand: Minimum demand (integer, before normalisation).
        max_demand: Maximum demand (integer, before normalisation).
        vehicle_capacity: Normalised vehicle capacity.
        capacity: Raw capacity used for normalisation.
        depot_type: Depot placement ("center", "corner", "random").

    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        min_demand: int = 1,
        max_demand: int = 10,
        vehicle_capacity: float = 1.0,
        capacity: Optional[float] = None,
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        rng: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise CVRPGenerator.



        Args:
            num_loc: Number of customer locations (excluding depot).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution ("uniform", "normal", "clustered").
            min_demand: Minimum integer demand (before normalisation).
            max_demand: Maximum integer demand (before normalisation).
            vehicle_capacity: Normalised vehicle capacity (default 1.0).
            capacity: Raw vehicle capacity used for normalisation.  If None,
                uses the lookup table from Kool et al. (2019).
            depot_type: Depot placement ("center", "corner", "random").
            device: Target device.
            rng: Optional numpy RNG.
            generator: Optional torch RNG.
            kwargs: Forwarded to Generator base.
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
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = vehicle_capacity
        self.depot_type = depot_type

        if capacity is None:
            closest = min(CAPACITIES.keys(), key=lambda x: abs(x - num_loc))
            capacity = CAPACITIES.get(num_loc, CAPACITIES[closest])
        self.capacity = capacity

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate a batch of CVRP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with:
                locs: [*B, num_loc, 2]  — customer coordinates
                depot: [*B, 2]           — depot coordinate
                demand: [*B, num_loc]     — normalised demand in (0, 1]
                vehicle_capacity: [*B]              — vehicle capacity (normalised to 1.0)
        """
        locs = self._generate_locations(batch_size)
        depot = self._generate_depot(batch_size)

        # Integer demands in [min_demand, max_demand], normalised by capacity
        raw_demand = torch.randint(
            self.min_demand,
            self.max_demand + 1,
            (*batch_size, self.num_loc),
            device=self.device,
            generator=self.generator,
        ).float()
        demand = raw_demand / self.capacity

        vehicle_capacity = torch.full((*batch_size,), self.vehicle_capacity, device=self.device, dtype=torch.float32)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "demand": demand,
                "vehicle_capacity": vehicle_capacity,
            },
            batch_size=batch_size,
            device=self.device,
        )

    def _generate_depot(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate depot location.

        Args:
            batch_size: Batch size.

        Returns:
            Depot location.
        """
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2.0
            return torch.full((*batch_size, 2), center, device=self.device, dtype=torch.float32)
        elif self.depot_type == "corner":
            return torch.full((*batch_size, 2), self.min_loc, device=self.device, dtype=torch.float32)
        else:  # random
            return self._uniform_locations(batch_size, 1).squeeze(-2)
