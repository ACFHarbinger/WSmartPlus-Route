"""
Data generators for combinatorial optimization problem instances.

This module provides generator classes that create problem instances
as TensorDict objects for use with RL4CO-style environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from logic.src.utils.data.data_utils import generate_waste_prize
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
        self.bins = kwargs.get("bins")
        self._kwargs = kwargs

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return all arguments used to initialize the generator."""
        # This is a helper to allow cloning/moving the generator
        # It should reconstruct the full kwargs including explicit args
        # But for now, let's just rely on subclasses implementing `to` or capturing enough in _kwargs?
        # Actually, subclasses assign args to self.
        return self._kwargs

    def to(self, device: Union[str, torch.device]) -> Generator:
        """
        Return a copy of the generator on the specified device.

        Args:
            device: Target device.

        Returns:
            New Generator instance on device.
        """
        # Get the class
        cls = self.__class__
        # Get all attributes that match __init__ signature?
        # This is hard to do generically without introspection or explicit saving.
        # Let's save `all_args` in `__init__` instead.

        # We can use the variables stored in self.
        # This requires knowing the mapping from init args to self attributes.
        # For Generator base:
        kwargs = self._kwargs.copy()
        kwargs.update(
            {
                "num_loc": self.num_loc,
                "min_loc": self.min_loc,
                "max_loc": self.max_loc,
                "loc_distribution": self.loc_distribution,
                "device": device,
            }
        )

        # For VRPP:
        if hasattr(self, "min_waste"):
            kwargs.update(
                {
                    "min_waste": self.min_waste,  # type: ignore[attr-defined]
                    "max_waste": self.max_waste,  # type: ignore[attr-defined]
                    "waste_distribution": self.waste_distribution,  # type: ignore[attr-defined]
                    "min_prize": self.min_prize,  # type: ignore[attr-defined]
                    "max_prize": self.max_prize,  # type: ignore[attr-defined]
                    "prize_distribution": self.prize_distribution,  # type: ignore[attr-defined]
                    "capacity": self.capacity,  # type: ignore[attr-defined]
                    "max_length": self.max_length,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )

        # For WCVRP:
        if hasattr(self, "min_fill"):
            kwargs.update(
                {
                    "min_fill": self.min_fill,  # type: ignore[attr-defined]
                    "max_fill": self.max_fill,  # type: ignore[attr-defined]
                    "fill_distribution": self.fill_distribution,  # type: ignore[attr-defined]
                    "capacity": self.capacity,  # type: ignore[attr-defined]
                    "cost_km": self.cost_km,  # type: ignore[attr-defined]
                    "revenue_kg": self.revenue_kg,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )

        # For SCWCVRP:
        if hasattr(self, "noise_mean"):
            kwargs.update(
                {
                    "noise_mean": self.noise_mean,  # type: ignore[attr-defined]
                    "noise_variance": self.noise_variance,  # type: ignore[attr-defined]
                }
            )

        return cls(**kwargs)

    @abstractmethod
    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """
        Generate a batch of problem instances.

        This method should be implemented by subclasses to create
        problem-specific instance data.

        Args:
            batch_size: Batch size(s) to generate.

        Returns:
            TensorDict containing the generated instances.
        """
        raise NotImplementedError

    def __call__(self, batch_size: Union[int, list[int], tuple[int, ...]] = 1) -> TensorDict:
        """
        Generate a batch of problem instances.

        Args:
            batch_size: Number of instances or batch size tuple to generate.

        Returns:
            TensorDict containing the generated instances.
        """
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        else:
            batch_size = tuple(batch_size)

        td = self._generate(batch_size)
        return td.to(self.device)

    def _generate_locations(self, batch_size: tuple[int, ...], num_loc: Optional[int] = None) -> torch.Tensor:
        """
        Generate location coordinates based on the specified distribution.

        Args:
            batch_size: Batch size tuple.
            num_loc: Number of locations (uses self.num_loc if None).

        Returns:
            Tensor of shape (*batch_size, num_loc, 2) containing coordinates.
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

    def _uniform_locations(self, batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor:
        """Generate uniformly distributed locations."""
        return torch.rand(*batch_size, num_loc, 2, device=self.device) * (self.max_loc - self.min_loc) + self.min_loc

    def _normal_locations(self, batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor:
        """Generate normally distributed locations (clipped to bounds)."""
        center = (self.max_loc + self.min_loc) / 2
        std = (self.max_loc - self.min_loc) / 6  # ~99.7% within bounds
        locs = torch.randn(*batch_size, num_loc, 2, device=self.device) * std + center
        return torch.clamp(locs, self.min_loc, self.max_loc)

    def _clustered_locations(self, batch_size: tuple[int, ...], num_loc: int, num_clusters: int = 3) -> torch.Tensor:
        """Generate clustered locations using Gaussian mixture."""
        num_clusters = self._kwargs.get("num_clusters", num_clusters)

        # Generate cluster centers
        centers = self._uniform_locations(batch_size, num_clusters)

        # Assign points to clusters
        cluster_assignments = torch.randint(0, num_clusters, (*batch_size, num_loc), device=self.device)

        # Generate points around cluster centers
        std = (self.max_loc - self.min_loc) / 10
        offsets = torch.randn(*batch_size, num_loc, 2, device=self.device) * std

        # Get assigned cluster centers
        locs = torch.zeros(*batch_size, num_loc, 2, device=self.device)
        for i in range(num_clusters):
            mask = cluster_assignments == i
            locs[mask] = centers[..., i : i + 1, :].expand(*batch_size, num_loc, -1)[mask]

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
        depot_type: str = "center",
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
            depot_type: Depot placement ("center", "corner", "random").
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, **kwargs)

        self.min_fill = min_fill
        self.max_fill = max_fill
        self.fill_distribution = kwargs.get("data_distribution", fill_distribution)
        self.capacity = capacity if capacity is not None else 100.0
        self.cost_km = cost_km
        self.revenue_kg = revenue_kg
        self.depot_type = depot_type

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate WCVRP instances."""
        # Generate locations
        locs = self._generate_locations(batch_size)

        # Generate depot
        depot = self._generate_depot(batch_size)

        # Generate fill levels (waste amount at each bin)
        fill = self._generate_fill_levels(batch_size)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "demand": fill,  # Fill level as demand
                "capacity": torch.full((*batch_size,), self.capacity, device=self.device),
                "max_waste": torch.full(
                    (*batch_size,), 1.0, device=self.device
                ),  # Consistent with definitions.MAX_WASTE
                "cost_km": torch.full((*batch_size,), self.cost_km, device=self.device),
                "revenue_kg": torch.full((*batch_size,), self.revenue_kg, device=self.device),
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

    def _generate_fill_levels(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate bin fill levels."""
        # Use common utility for consistency
        bs = batch_size[0] if batch_size else 1
        coords = (self._generate_depot(batch_size), self._generate_locations(batch_size))
        fill = generate_waste_prize(self.num_loc, self.fill_distribution, coords, bs, bins=self.bins)
        if isinstance(fill, np.ndarray):
            fill = torch.from_numpy(fill).float()
        return fill.to(self.device).view(*batch_size, self.num_loc)


class SCWCVRPGenerator(WCVRPGenerator):
    """
    Generator for Stochastic Capacitated Waste Collection VRP (SCWCVRP) instances.

    Adds noise to the fill levels to simulate uncertain demand.
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
        depot_type: str = "center",
        noise_mean: float = 0.0,
        noise_variance: float = 0.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize SCWCVRP generator."""
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            min_fill=min_fill,
            max_fill=max_fill,
            fill_distribution=fill_distribution,
            capacity=capacity,
            cost_km=cost_km,
            revenue_kg=revenue_kg,
            depot_type=depot_type,
            device=device,
            **kwargs,
        )
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate SCWCVRP instances."""
        td = super()._generate(batch_size)

        # Rename 'demand' to 'real_waste' (internal) and add 'noisy_waste'
        real_waste = td["demand"].clone()
        td["real_waste"] = real_waste

        if self.noise_variance > 0:
            noise = torch.normal(
                mean=self.noise_mean,
                std=self.noise_variance**0.5,
                size=real_waste.size(),
                device=self.device,
            )
            noisy_waste = (real_waste + noise).clamp(min=0.0, max=1.0)
        else:
            noisy_waste = real_waste.clone()

        td["waste"] = noisy_waste
        td["demand"] = noisy_waste  # Set demand to the noisy version for the agent

        return td


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


class JSSPGenerator(Generator):
    """
    Generator for Job Shop Scheduling Problem (JSSP) instances.

    Generates random JSSP instances where each job consists of M operations
    to be processed on M different machines in a specific order.

    Features:
    - proc_time: Processing time for each operation (Job j, Op i)
    - machine_order: Machine required for each operation (Job j, Op i) -> Machine ID
    """

    def __init__(
        self,
        num_jobs: int = 10,
        num_machines: int = 10,
        min_duration: int = 1,
        max_duration: int = 99,
        duration_distribution: str = "uniform",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize JSSP generator.

        Args:
            num_jobs: Number of jobs (J).
            num_machines: Number of machines (M).
            min_duration: Minimum processing time.
            max_duration: Maximum processing time.
            duration_distribution: Distribution for durations ("uniform").
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_loc=num_jobs,  # Reusing num_loc to store num_jobs if needed, or ignored
            min_loc=min_duration,
            max_loc=max_duration,
            loc_distribution=duration_distribution,
            device=device,
            **kwargs,
        )
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.min_duration = min_duration
        self.max_duration = max_duration

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate JSSP instances."""
        # Random durations
        proc_time = torch.randint(
            self.min_duration,
            self.max_duration + 1,
            (*batch_size, self.num_jobs, self.num_machines),
            device=self.device,
            dtype=torch.float32,
        )

        # Generate machine order: (batch, J, M)
        # For each job, the sequence of machines is a random permutation of 0..M-1
        # We need to generate a permutation for each job in the batch

        # 1. Create base indices [0, ..., M-1]
        # 2. Shuffle them for each job

        # shape: (batch * J, M)
        raw_perms = torch.rand((*batch_size, self.num_jobs, self.num_machines), device=self.device).argsort(dim=-1)
        machine_order = raw_perms

        return TensorDict(
            {
                "proc_time": proc_time,
                "machine_order": machine_order,
                "num_jobs": torch.full((*batch_size,), self.num_jobs, device=self.device),
                "num_machines": torch.full((*batch_size,), self.num_machines, device=self.device),
            },
            batch_size=batch_size,
            device=self.device,
        )


# Registry of available generators
GENERATOR_REGISTRY: dict[str, type[Generator]] = {
    "vrpp": VRPPGenerator,
    "cvrpp": VRPPGenerator,  # Same generator, different env handles capacity
    "wcvrp": WCVRPGenerator,
    "cwcvrp": WCVRPGenerator,
    "scwcvrp": SCWCVRPGenerator,
    "tsp": TSPGenerator,
    "pdp": PDPGenerator,
    "jssp": JSSPGenerator,
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
        raise ValueError(f"Unknown generator: {name}. Available generators: {list(GENERATOR_REGISTRY.keys())}")
    return GENERATOR_REGISTRY[name](**kwargs)
