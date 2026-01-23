"""
Legacy problem definitions for backward compatibility.
Provides the BaseProblem interface expected by legacy AttentionModel and Evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.utils.functions.beam_search import beam_search as beam_search_func


class BaseProblem:
    """
    Legacy base class for routing problems.
    """

    @staticmethod
    def validate_tours(pi: torch.Tensor) -> bool:
        """Validates tours (no duplicates except depot)."""
        if pi.size(-1) == 1:
            assert (pi == 0).all()
            return True
        sorted_pi: torch.Tensor = pi.data.sort(1)[0]
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all()
        return True

    @staticmethod
    def get_tour_length(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates tour length."""
        if pi.size(-1) == 1:
            return torch.zeros(pi.size(0), device=pi.device)

        use_dist_matrix = dist_matrix is not None and isinstance(dist_matrix, torch.Tensor)
        if use_dist_matrix:
            # Simple distance matrix lookup
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)
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
                    0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0
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
    def beam_search(cls, input, beam_size, cost_weights, model=None, **kwargs):
        """Beam search bridge."""
        assert model is not None
        fixed = model.precompute_fixed(input, edges=input.get("edges"))

        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, normalize=True)

        # Note: make_state is problem-specific, must be implemented by subclasses
        state = cls.make_state(input, cost_weights=cost_weights, **kwargs)
        return beam_search_func(state, beam_size, propose_expansions)

    @staticmethod
    def make_state(*args, **kwargs):
        """Should be overridden."""
        raise NotImplementedError


class VRPP(BaseProblem):
    NAME = "vrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        VRPP.validate_tours(pi)
        if pi.size(-1) == 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return z, {"total": z}, None

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        w = waste_with_depot.gather(1, pi)
        if "max_waste" in dataset:
            w = w.clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)
        length = VRPP.get_tour_length(dataset, pi, dist_matrix)

        cost_km = dataset.get("cost_km", 1.0)
        revenue_kg = dataset.get("revenue_kg", 0.1625)

        neg_profit = length * cost_km - waste * revenue_kg
        if cw_dict is not None:
            neg_profit = cw_dict.get("length", 1.0) * length * cost_km - cw_dict.get("waste", 1.0) * waste * revenue_kg

        return neg_profit, {"length": length, "waste": waste, "total": neg_profit}, None


class CVRPP(VRPP):
    NAME = "cvrpp"


class WCVRP(BaseProblem):
    NAME = "wcvrp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        WCVRP.validate_tours(pi)
        if pi.size(-1) == 1:
            overflows = (dataset["waste"] >= dataset.get("max_waste", 1.0)).float().sum(-1)
            return overflows, {"total": overflows}, None

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        visited_mask = torch.zeros_like(waste_with_depot, dtype=torch.bool)
        visited_mask.scatter_(1, pi, True)

        max_w = dataset.get("max_waste", 1.0)
        overflow_mask = waste_with_depot >= max_w
        overflows = torch.sum(overflow_mask[:, 1:] & ~visited_mask[:, 1:], dim=-1).float()

        length = WCVRP.get_tour_length(dataset, pi, dist_matrix)
        waste = waste_with_depot.gather(1, pi).clamp(max=max_w).sum(dim=-1)

        cost = overflows + length - waste
        if cw_dict is not None:
            cost = (
                cw_dict.get("overflows", 1.0) * overflows
                + cw_dict.get("length", 1.0) * length
                - cw_dict.get("waste", 1.0) * waste
            )

        return cost, {"overflows": overflows, "length": length, "waste": waste, "total": cost}, None


class CWCVRP(WCVRP):
    NAME = "cwcvrp"


class SDWCVRP(WCVRP):
    NAME = "sdwcvrp"


class SCWCVRP(WCVRP):
    NAME = "scwcvrp"
