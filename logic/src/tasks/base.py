"""
Base class for optimization problem definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import torch

from logic.src.utils.functions.beam_search import beam_search as beam_search_func
from logic.src.utils.functions.boolmask import mask_long2bool


class BaseProblem:
    """
    Base class for routing problems (WCVRP, VRPP, SCWCVRP).
    Consolidates shared logic for cost calculation, tour validation, and beam search.
    """

    @staticmethod
    def validate_tours(pi: torch.Tensor) -> bool:
        """
        Validates that tours are valid (contain 0 to n-1, no duplicates except depot).

        Args:
            pi: The sequence of visited nodes (batch_size, tour_length).
        """
        if pi.size(-1) == 1:
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            return True

        sorted_pi: torch.Tensor = pi.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicate nodes found in tour"
        return True

    @staticmethod
    def get_tour_length(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the tour length (L2 norm or from distance matrix).

        Args:
            dataset: Dataset containing 'depot', 'loc', etc.
            pi: tour sequence.
            dist_matrix: Precomputed distance matrix.

        Returns:
            length of each tour in the batch.
        """
        if pi.size(-1) == 1:
            return torch.zeros(pi.size(0), device=pi.device)

        # Check if dist_matrix is valid (not None and is a real tensor)
        use_dist_matrix = dist_matrix is not None and isinstance(dist_matrix, torch.Tensor)

        if use_dist_matrix:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask: torch.Tensor = dst_vertices != 0
            pair_mask: torch.Tensor = (src_vertices != 0) & (dst_mask)
            dists: torch.Tensor = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()

            last_dst: torch.Tensor = torch.max(
                dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
                dim=1,
            ).indices

            length: torch.Tensor = (
                dist_matrix[
                    0,
                    dst_vertices[
                        torch.arange(dst_vertices.size(0), device=dst_vertices.device),
                        last_dst,
                    ],
                    0,
                ]
                + dists.sum(dim=1)
                + dist_matrix[0, 0, pi[:, 0]]
            )
        else:
            loc_with_depot: torch.Tensor = torch.cat((dataset["depot"][:, None, :], dataset["loc"]), 1)
            d: torch.Tensor = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)
                + (d[:, 0] - dataset["depot"]).norm(p=2, dim=-1)
                + (d[:, -1] - dataset["depot"]).norm(p=2, dim=-1)
            )
        return length

    @classmethod
    def beam_search(
        cls,
        input: Dict[str, Any],
        beam_size: int,
        cost_weights: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        expand_size: Optional[int] = None,
        compress_mask: bool = False,
        model: Optional[Any] = None,
        max_calc_batch_size: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Standardized beam search implementation.
        """
        assert model is not None, "Provide model"
        fixed: Dict[str, Any] = model.precompute_fixed(input)

        def propose_expansions(beam: Any) -> Any:
            """
            Proposes next nodes for the beam based on model scores.
            """
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state: Any = cls.make_state(  # type: ignore
            input,
            edges,
            cost_weights,
            visited_dtype=torch.int64 if compress_mask else torch.uint8,
            **kwargs,
        )
        return beam_search_func(state, beam_size, propose_expansions)


class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for routing datasets.
    """

    def __init__(self) -> None:
        """Initializes the dataset."""
        super(BaseDataset, self).__init__()
        self.data: List[Dict[str, Any]] = []
        self.size: int = 0

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the data sample at the given index."""
        return self.data[idx]

    def __setitem__(self, key: str, values: List[Any]) -> None:
        """Updates a specific field across all samples in the dataset."""

        def __update_item(inst: Dict[str, Any], k: str, v: Any) -> Dict[str, Any]:
            """Internal helper to update a dictionary item."""
            inst[k] = v
            return inst

        self.data = [__update_item(x, key, val) for x, val in zip(self.data, values)]


class BaseState:
    """
    Collection of common state functionality.
    To be injected into NamedTuple classes.
    """

    visited_: torch.Tensor
    coords: torch.Tensor
    i: torch.Tensor
    prev_a: torch.Tensor
    edges: torch.Tensor

    @property
    def visited(self) -> Optional[torch.Tensor]:
        """Returns a boolean mask of visited nodes."""
        if not hasattr(self, "visited_"):
            return None
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self) -> torch.Tensor:
        """Calculates the Euclidean distance matrix between all coordinates."""
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def all_finished(self) -> bool:
        """Checks if all instances in the batch have returned to the depot."""
        return bool(self.i.item() > 0 and (self.prev_a == 0).all())

    def get_current_node(self) -> torch.Tensor:
        """Returns the current node index."""
        return self.prev_a

    def get_edges_mask(self) -> torch.Tensor:
        """Returns a mask based on graph edges for the current node."""
        batch_size, n_coords, _ = self.coords.size()
        if self.i.item() == 0:
            return torch.zeros(batch_size, 1, n_coords, dtype=torch.uint8, device=self.coords.device)
        else:
            return self.edges.gather(1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_coords))

    def get_edges(self) -> torch.Tensor:
        """Returns the graph edge indices."""
        return self.edges

    def construct_solutions(self, actions: Any) -> Any:
        """Returns the sequences of actions as solutions."""
        return actions

    @staticmethod
    def initialize_common(input: Dict[str, torch.Tensor], visited_dtype: torch.dtype = torch.uint8) -> Dict[str, Any]:
        """
        Computes common initialization fields.
        """
        depot: torch.Tensor = input["depot"]
        loc: torch.Tensor = input["loc"]
        batch_size, n_loc, _ = loc.size()
        coords: torch.Tensor = torch.cat((depot[:, None, :], loc), -2)

        ids: torch.Tensor = torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None]
        prev_a: torch.Tensor = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)

        visited_: torch.Tensor
        if visited_dtype == torch.uint8:
            visited_ = torch.zeros(batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device)
        else:
            visited_ = torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)

        lengths: torch.Tensor = torch.zeros(batch_size, 1, device=loc.device)
        cur_coord: torch.Tensor = depot[:, None, :]
        i: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=loc.device)

        return {
            "coords": coords,
            "ids": ids,
            "prev_a": prev_a,
            "visited_": visited_,
            "lengths": lengths,
            "cur_coord": cur_coord,
            "i": i,
            "batch_size": batch_size,
            "n_loc": n_loc,
        }


def refactor_state(cls: Type[Any]) -> Type[Any]:
    """
    Decorator to inject BaseState methods into a NamedTuple class.
    """
    for name, attr in BaseState.__dict__.items():
        if not name.startswith("__") and name != "initialize_common":
            setattr(cls, name, attr)
    return cls
