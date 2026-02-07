"""
VRPP and CVRPP problem definitions.
"""

import torch

from logic.src.constants.tasks import COST_KM, REVENUE_KG
from logic.src.envs.tasks.base import BaseProblem


class VRPP(BaseProblem):
    """
    Vehicle Routing Problem with Profits (VRPP).
    Objective: Maximize Profit (Revenue - Cost).
    """

    NAME = "vrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute VRPP costs/rewards.

        Args:
            dataset: Problem data.
            pi: Tours [batch, nodes].
            cw_dict: Cost weights dictionary.
            dist_matrix: Optional distance matrix.

        Returns:
            Tuple of (negative_profit, cost_dict, None).
        """
        VRPP.validate_tours(pi)
        if pi.size(-1) == 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return (
                z,
                {"length": z, "waste": z, "overflows": z, "total": z},
                None,
            )

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        w = waste_with_depot.gather(1, pi)
        if "max_waste" in dataset.keys():
            w = w.clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)
        length = VRPP.get_tour_length(dataset, pi, dist_matrix)

        cost_km = dataset.get("cost_km", COST_KM)
        revenue_kg = dataset.get("revenue_kg", REVENUE_KG)

        neg_profit = length * cost_km - waste * revenue_kg
        if cw_dict is not None:
            neg_profit = cw_dict.get("length", 1.0) * length * cost_km - cw_dict.get("waste", 1.0) * waste * revenue_kg

        return (
            neg_profit,
            {"length": length, "waste": waste, "overflows": torch.zeros_like(neg_profit), "total": neg_profit},
            None,
        )
