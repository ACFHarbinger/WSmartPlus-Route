"""
Data generators for combinatorial optimization problem instances.

This module provides generator classes that create problem instances
as TensorDict objects for use with RL4CO-style environments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import torch
from tensordict import TensorDict


class Generator(ABC):
    """
    Abstract base class for problem instance generators.

    Generators create batches of problem instances as TensorDict objects,
    supporting various probability distributions for feature generation.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the generator.

        Args:
            num_loc: Number of locations (excluding depot).
            min_loc: Minimum coordinate value.
            max_loc: Maximum coordinate value.
            loc_distribution: Distribution for location generation.
                Can be "uniform", "normal", "clustered", or a callable.
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_distribution = loc_distribution
        self.device = torch.device(device)
        self._kwargs = kwargs

    @abstractmethod
    def _generate(self, batch_size: int) -> TensorDict:
        """
        Generate a batch of problem instances.

        This method should be implemented by subclasses to create
        problem-specific instance data.

        Args:
            batch_size: Number of instances to generate.

        Returns:
            TensorDict containing the generated instances.
        """
        raise NotImplementedError

    def __call__(self, batch_size: int) -> TensorDict:
        """
        Generate a batch of problem instances.

        Args:
            batch_size: Number of instances to generate.

        Returns:
            TensorDict containing the generated instances.
        """
        td = self._generate(batch_size)
        return td.to(self.device)

    def _generate_locations(self, batch_size: int, num_loc: Optional[int] = None) -> torch.Tensor:
        """
        Generate location coordinates based on the specified distribution.

        Args:
            batch_size: Number of instances.
            num_loc: Number of locations (uses self.num_loc if None).

        Returns:
            Tensor of shape (batch_size, num_loc, 2) containing coordinates.
        """
        num_loc = num_loc or self.num_loc

        if callable(self.loc_distribution):
            return self.loc_distribution(batch_size, num_loc)

        if self.loc_distribution == "uniform":
            return self._uniform_locations(batch_size, num_loc)
        elif self.loc_distribution == "normal":
            return self._normal_locations(batch_size, num_loc)
        elif self.loc_distribution == "clustered":
            return self._clustered_locations(batch_size, num_loc)
        else:
            raise ValueError(f"Unknown location distribution: {self.loc_distribution}")

    def _uniform_locations(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Generate uniformly distributed locations."""
        return torch.rand(batch_size, num_loc, 2, device=self.device) * (self.max_loc - self.min_loc) + self.min_loc

    def _normal_locations(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Generate normally distributed locations (clipped to bounds)."""
        center = (self.max_loc + self.min_loc) / 2
        std = (self.max_loc - self.min_loc) / 6  # ~99.7% within bounds
        locs = torch.randn(batch_size, num_loc, 2, device=self.device) * std + center
        return torch.clamp(locs, self.min_loc, self.max_loc)

    def _clustered_locations(self, batch_size: int, num_loc: int, num_clusters: int = 3) -> torch.Tensor:
        """Generate clustered locations using Gaussian mixture."""
        num_clusters = self._kwargs.get("num_clusters", num_clusters)

        # Generate cluster centers
        centers = self._uniform_locations(batch_size, num_clusters)

        # Assign points to clusters
        cluster_assignments = torch.randint(0, num_clusters, (batch_size, num_loc), device=self.device)

        # Generate points around cluster centers
        std = (self.max_loc - self.min_loc) / 10
        offsets = torch.randn(batch_size, num_loc, 2, device=self.device) * std

        # Get assigned cluster centers
        locs = torch.zeros(batch_size, num_loc, 2, device=self.device)
        for i in range(num_clusters):
            mask = cluster_assignments == i
            locs[mask] = centers[:, i : i + 1, :].expand(-1, num_loc, -1)[mask]

        locs = locs + offsets
        return torch.clamp(locs, self.min_loc, self.max_loc)


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
        self.waste_distribution = waste_distribution
        self.min_prize = min_prize
        self.max_prize = max_prize
        self.prize_distribution = prize_distribution
        self.capacity = capacity
        self.max_length = max_length
        self.depot_type = depot_type

    def _generate(self, batch_size: int) -> TensorDict:
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
        max_waste = torch.full((batch_size,), self.capacity, device=self.device)

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
                "capacity": torch.full((batch_size,), self.capacity, device=self.device),
                "max_waste": max_waste,
                "max_length": torch.full((batch_size,), max_length, device=self.device),
            },
            batch_size=[batch_size],
            device=self.device,
        )

    def _generate_depot(self, batch_size: int) -> torch.Tensor:
        """Generate depot location based on depot_type."""
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2
            return torch.full((batch_size, 2), center, device=self.device)
        elif self.depot_type == "corner":
            return torch.full((batch_size, 2), self.min_loc, device=self.device)
        elif self.depot_type == "random":
            return self._uniform_locations(batch_size, 1).squeeze(1)
        else:
            raise ValueError(f"Unknown depot type: {self.depot_type}")

    def _generate_waste(self, batch_size: int) -> torch.Tensor:
        """Generate waste/demand values."""
        if self.waste_distribution == "uniform":
            return (
                torch.rand(batch_size, self.num_loc, device=self.device) * (self.max_waste - self.min_waste)
                + self.min_waste
            )
        elif self.waste_distribution == "gamma":
            # Gamma distribution for more realistic waste patterns
            alpha = self._kwargs.get("waste_alpha", 2.0)
            beta = self._kwargs.get("waste_beta", 0.3)
            waste = torch.distributions.Gamma(alpha, 1 / beta).sample((batch_size, self.num_loc))
            waste = waste.to(self.device)
            return torch.clamp(waste, self.min_waste, self.max_waste)
        else:
            raise ValueError(f"Unknown waste distribution: {self.waste_distribution}")

    def _generate_prize(self, batch_size: int) -> torch.Tensor:
        """Generate prize values."""
        if self.prize_distribution == "uniform":
            return (
                torch.rand(batch_size, self.num_loc, device=self.device) * (self.max_prize - self.min_prize)
                + self.min_prize
            )
        elif self.prize_distribution == "distance_correlated":
            # Prize correlated with distance from depot
            # (further locations have higher prizes as incentive)
            depot = self._generate_depot(batch_size)
            locs = self._generate_locations(batch_size)
            distances = torch.norm(locs - depot.unsqueeze(1), dim=-1)
            max_dist = distances.max(dim=-1, keepdim=True).values
            normalized_dist = distances / (max_dist + 1e-8)
            return normalized_dist * (self.max_prize - self.min_prize) + self.min_prize
        else:
            raise ValueError(f"Unknown prize distribution: {self.prize_distribution}")


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
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, **kwargs)

        self.min_fill = min_fill
        self.max_fill = max_fill
        self.fill_distribution = fill_distribution
        self.capacity = capacity
        self.cost_km = cost_km
        self.revenue_kg = revenue_kg

    def _generate(self, batch_size: int) -> TensorDict:
        """Generate WCVRP instances."""
        # Generate locations
        locs = self._generate_locations(batch_size)

        # Generate depot at center
        depot = torch.full((batch_size, 2), (self.max_loc + self.min_loc) / 2, device=self.device)

        # Generate fill levels (waste amount at each bin)
        fill = self._generate_fill_levels(batch_size)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "demand": fill,  # Fill level as demand
                "capacity": torch.full((batch_size,), self.capacity, device=self.device),
                "cost_km": torch.full((batch_size,), self.cost_km, device=self.device),
                "revenue_kg": torch.full((batch_size,), self.revenue_kg, device=self.device),
            },
            batch_size=[batch_size],
            device=self.device,
        )

    def _generate_fill_levels(self, batch_size: int) -> torch.Tensor:
        """Generate bin fill levels."""
        if self.fill_distribution == "uniform":
            return (
                torch.rand(batch_size, self.num_loc, device=self.device) * (self.max_fill - self.min_fill)
                + self.min_fill
            )
        elif self.fill_distribution == "beta":
            # Beta distribution for fill levels (tends toward 0 or 1)
            alpha = self._kwargs.get("fill_alpha", 2.0)
            beta = self._kwargs.get("fill_beta", 5.0)
            fill = torch.distributions.Beta(alpha, beta).sample((batch_size, self.num_loc))
            fill = fill.to(self.device)
            return fill * (self.max_fill - self.min_fill) + self.min_fill
        else:
            raise ValueError(f"Unknown fill distribution: {self.fill_distribution}")


class TSPGenerator(Generator):
    """
    Generator for Traveling Salesman Problem (TSP) instances.

    Simple generator for TSP benchmarking with just locations.
    """

    def _generate(self, batch_size: int) -> TensorDict:
        """Generate TSP instances."""
        # Generate locations (including depot as first location)
        all_locs = self._generate_locations(batch_size, self.num_loc + 1)

        return TensorDict(
            {
                "locs": all_locs[:, 1:, :],  # Customer locations
                "depot": all_locs[:, 0, :],  # First location as depot
            },
            batch_size=[batch_size],
            device=self.device,
        )


# Registry of available generators
GENERATOR_REGISTRY: dict[str, type[Generator]] = {
    "vrpp": VRPPGenerator,
    "cvrpp": VRPPGenerator,  # Same generator, different env handles capacity
    "wcvrp": WCVRPGenerator,
    "cwcvrp": WCVRPGenerator,
    "tsp": TSPGenerator,
}


def get_generator(name: str, **kwargs: Any) -> Generator:
    """
    Get a generator by name.

    Args:
        name: Generator name (e.g., "vrpp", "wcvrp", "tsp").
        **kwargs: Generator configuration parameters.

    Returns:
        Configured Generator instance.

    Raises:
        ValueError: If generator name is not found.
    """
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator: {name}. " f"Available generators: {list(GENERATOR_REGISTRY.keys())}")
    return GENERATOR_REGISTRY[name](**kwargs)
