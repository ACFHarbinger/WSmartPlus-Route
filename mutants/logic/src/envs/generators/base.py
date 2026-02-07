"""
Base generator class for problem instances.
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
        self.bins = kwargs.get("bins")
        self._kwargs = kwargs

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return all arguments used to initialize the generator."""
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

        # We can use the variables stored in self.
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
                    "min_waste": getattr(self, "min_waste"),
                    "max_waste": getattr(self, "max_waste"),
                    "waste_distribution": getattr(self, "waste_distribution"),
                    "min_prize": getattr(self, "min_prize"),
                    "max_prize": getattr(self, "max_prize"),
                    "prize_distribution": getattr(self, "prize_distribution"),
                    "capacity": getattr(self, "capacity"),
                    "max_length": getattr(self, "max_length"),
                    "depot_type": getattr(self, "depot_type"),
                }
            )

        # For WCVRP:
        if hasattr(self, "min_fill"):
            kwargs.update(
                {
                    "min_fill": getattr(self, "min_fill"),
                    "max_fill": getattr(self, "max_fill"),
                    "fill_distribution": getattr(self, "fill_distribution"),
                    "capacity": getattr(self, "capacity"),
                    "cost_km": getattr(self, "cost_km"),
                    "revenue_kg": getattr(self, "revenue_kg"),
                    "depot_type": getattr(self, "depot_type"),
                }
            )

        # For SCWCVRP:
        if hasattr(self, "noise_mean"):
            kwargs.update(
                {
                    "noise_mean": getattr(self, "noise_mean"),
                    "noise_variance": getattr(self, "noise_variance"),
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
