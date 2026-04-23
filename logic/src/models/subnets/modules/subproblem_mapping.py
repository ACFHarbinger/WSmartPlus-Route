"""GLOP Subproblem Mapping.

This module provides the SubproblemMapping dataclass, which tracks the relationship
between local subproblem node indices and their global counterparts in the
original large-scale problem.

Attributes:
    SubproblemMapping: Data structure for tracking subproblem indices and coordinates.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.subproblem_mapping import SubproblemMapping
    >>> mapping = SubproblemMapping(0, 0, torch.randn(5, 2), torch.arange(5), 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SubproblemMapping:
    """Mapping from local subproblem space to global problem space.

    A container used during subproblem-based decomposition (e.g., in GLOP) to
    maintain traceability between partitioned clusters and the parent environment.

    Attributes:
        batch_idx (int): Global index of the specific instance in the current batch.
        sample_idx (int): Local index if multiple paths (beams) per instance exist.
        subprob_coordinates (torch.Tensor): Coordinates of the nodes in this subproblem.
        original_indices (torch.Tensor): Mapping of local node IDs to global scene IDs.
        route_idx (int): The identifier of the route being refined within the instance.
    """

    batch_idx: int
    sample_idx: int
    subprob_coordinates: torch.Tensor
    original_indices: torch.Tensor
    route_idx: int
