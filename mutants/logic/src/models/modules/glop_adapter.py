"""
GLOP Subproblem Adapters.

Convert global partitions to local subproblems for hierarchical solving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
from tensordict import TensorDict


@dataclass
class SubproblemMapping:
    """Mapping from subproblem to original problem indices."""

    batch_idx: int
    sample_idx: int
    subprob_coordinates: torch.Tensor
    original_indices: torch.Tensor
    route_idx: int


class SubproblemAdapter(ABC):
    """
    Base class for adapters that convert global partitions to local subproblems.

    GLOP uses a two-stage approach:
    1. Global: Partition nodes into groups using NAR policy
    2. Local: Solve each partition as independent subproblem

    This adapter handles the partition → subproblem conversion and
    solution merging.
    """

    subproblem_env_name: str = "tsp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
    ) -> None:
        """
        Initialize adapter with problem instance and partition.

        Args:
            td: TensorDict with problem data.
            partition_actions: Partition assignments (batch, num_nodes).
            subprob_batch_size: Batch size for subproblem solving.
        """
        self.td = td
        self.partition_actions = partition_actions
        self.subprob_batch_size = subprob_batch_size

        # Store original coordinates
        self.coordinates = td.get("locs", td.get("locations"))
        self.batch_size = self.coordinates.size(0)

        # Initialize storage for final actions
        self.final_actions = torch.zeros_like(partition_actions)

    @abstractmethod
    def get_batched_subprobs(self) -> Iterator[SubproblemMapping]:
        """
        Yield batches of subproblems for solving.

        Returns:
            Iterator of SubproblemMapping containing coordinates and indices.
        """
        pass

    @abstractmethod
    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """
        Update final actions with solved subproblem.

        Args:
            mapping: Mapping from subproblem to original indices.
            subprob_actions: Solution for the subproblem.
        """
        pass

    def get_actions(self) -> torch.Tensor:
        """Get final actions after all subproblems solved."""
        return self.final_actions


class TSPAdapter(SubproblemAdapter):
    """
    TSP Adapter for converting partitions to TSP subproblems.

    Each partition group becomes an independent TSP instance.
    """

    subproblem_env_name: str = "tsp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
        **kwargs,
    ) -> None:
        super().__init__(td, partition_actions, subprob_batch_size)

        # Extract subproblems from partitions
        self._subproblems: List[SubproblemMapping] = []
        self._extract_subproblems()

    def _extract_subproblems(self) -> None:
        """Extract TSP subproblems from partition actions."""
        for batch_idx in range(self.batch_size):
            coords = self.coordinates[batch_idx]
            actions = self.partition_actions[batch_idx]

            # Group nodes by partition
            unique_partitions = actions.unique()

            for route_idx, partition_id in enumerate(unique_partitions):
                # Get nodes in this partition
                mask = actions == partition_id
                indices = torch.where(mask)[0]

                if len(indices) < 2:
                    continue

                # Get coordinates for this partition
                subprob_coords = coords[indices]

                self._subproblems.append(
                    SubproblemMapping(
                        batch_idx=batch_idx,
                        sample_idx=0,
                        subprob_coordinates=subprob_coords,
                        original_indices=indices,
                        route_idx=route_idx,
                    )
                )

    def get_batched_subprobs(self) -> Iterator[SubproblemMapping]:
        """Yield subproblems in batches."""
        batch = []
        for subprob in self._subproblems:
            batch.append(subprob)
            if len(batch) >= self.subprob_batch_size:
                for item in batch:
                    yield item
                batch = []

        # Yield remaining
        for item in batch:
            yield item

    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """Update final actions with TSP solution."""
        # Map subproblem solution back to original indices
        original_indices = mapping.original_indices
        batch_idx = mapping.batch_idx

        # The subprob_actions gives tour order within the subproblem
        # We need to translate this to the original action format
        for pos, node_idx in enumerate(subprob_actions):
            if node_idx < len(original_indices):
                original_node = original_indices[node_idx]
                self.final_actions[batch_idx, pos] = original_node


class VRPAdapter(SubproblemAdapter):
    """
    VRP Adapter for converting partitions to VRP-style subproblems.

    Handles capacity constraints and depot visits.
    """

    subproblem_env_name: str = "cvrp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
        capacity: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(td, partition_actions, subprob_batch_size)

        self.capacity = capacity or td.get("vehicle_capacity", torch.tensor(1.0)).item()
        self._subproblems: List[SubproblemMapping] = []
        self._extract_subproblems()

    def _extract_subproblems(self) -> None:
        """Extract VRP subproblems from partition actions."""
        for batch_idx in range(self.batch_size):
            coords = self.coordinates[batch_idx]
            actions = self.partition_actions[batch_idx]

            # First node is depot
            depot = coords[0:1]

            # Group nodes by route assignment
            unique_routes = actions.unique()

            for route_idx, route_id in enumerate(unique_routes):
                mask = actions == route_id
                indices = torch.where(mask)[0]

                # Skip depot-only routes
                if len(indices) < 1:
                    continue

                # Include depot at start
                subprob_coords = torch.cat([depot, coords[indices]], dim=0)

                self._subproblems.append(
                    SubproblemMapping(
                        batch_idx=batch_idx,
                        sample_idx=0,
                        subprob_coordinates=subprob_coords,
                        original_indices=indices,
                        route_idx=route_idx,
                    )
                )

    def get_batched_subprobs(self) -> Iterator[SubproblemMapping]:
        """Yield subproblems in batches."""
        for subprob in self._subproblems:
            yield subprob

    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """Update final actions with VRP route solution."""
        original_indices = mapping.original_indices
        batch_idx = mapping.batch_idx

        # Map subproblem actions (which include depot) back to original
        for pos, node_idx in enumerate(subprob_actions):
            if node_idx == 0:
                # Depot - use 0 in final actions
                self.final_actions[batch_idx, pos] = 0
            elif node_idx <= len(original_indices):
                original_node = original_indices[node_idx - 1]
                self.final_actions[batch_idx, pos] = original_node


# Adapter registry
ADAPTER_REGISTRY = {
    "tsp": TSPAdapter,
    "cvrp": VRPAdapter,
    "vrpp": TSPAdapter,  # VRP variants can use TSP for subproblems
    "wcvrp": VRPAdapter,
}


def get_adapter(env_name: str) -> type:
    """Get adapter class for environment."""
    if env_name in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[env_name]
    # Default to TSP for unknown environments
    return TSPAdapter
