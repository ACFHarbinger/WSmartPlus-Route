"""
Waste Collection Vehicle Routing Problem (WCVRP) and its variants.
"""

import os
import pickle

import torch
from tqdm import tqdm

from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.network import apply_edges, compute_distance_matrix
from logic.src.utils.data_utils import generate_waste_prize, load_focus_coords
from logic.src.utils.definitions import MAX_WASTE, VEHICLE_CAPACITY
from logic.src.utils.problem_utils import calculate_edges, make_instance_generic

from ..base import BaseDataset, BaseProblem
from .state_cwcvrp import StateCWCVRP
from .state_sdwcvrp import StateSDWCVRP
from .state_wcvrp import StateWCVRP


class WCVRP(BaseProblem):
    """
    The Waste Collection Vehicle Routing Problem (WCVRP).

    In this problem, the goal is to find a route that minimizes a combination
    of overflows, tour length, and collected waste.
    """

    NAME = "wcvrp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the total cost for a set of tours based on overflows, length, and waste.
        """
        WCVRP.validate_tours(pi)

        if pi.size(-1) == 1:
            overflow_mask = dataset["waste"] >= dataset["max_waste"][:, None]
            overflows = torch.sum(overflow_mask, dim=-1, dtype=torch.float)
            cost = overflows if cw_dict is None else cw_dict["overflows"] * overflows
            c_dict = {
                "overflows": overflows,
                "length": torch.zeros_like(overflows).to(pi.device),
                "waste": torch.zeros_like(overflows).to(pi.device),
                "total": cost,
            }
            return cost, c_dict, None

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        w = waste_with_depot.gather(1, pi).clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)

        overflow_mask = waste_with_depot >= dataset["max_waste"][:, None]
        visited_mask = torch.zeros_like(waste_with_depot, dtype=torch.bool)
        visited_mask.scatter_(1, pi, True)
        overflows = torch.sum(overflow_mask[:, 1:] & ~visited_mask[:, 1:], dim=-1)

        length = WCVRP.get_tour_length(dataset, pi, dist_matrix)

        cost = (
            overflows + length - waste
            if cw_dict is None
            else cw_dict["overflows"] * overflows + cw_dict["length"] * length - cw_dict["waste"] * waste
        )
        c_dict = {
            "overflows": overflows,
            "length": length,
            "waste": waste,
            "total": cost,
        }
        return cost, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """Creates a WCVRP dataset."""
        return WCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """Initializes the WCVRP state."""
        profit_vars = kwargs.pop("profit_vars", None)
        vehicle_capacity = (
            profit_vars.get("vehicle_capacity", VEHICLE_CAPACITY) if profit_vars is not None else VEHICLE_CAPACITY
        )
        return StateWCVRP.initialize(*args, vehicle_capacity=vehicle_capacity, **kwargs)


class CWCVRP(BaseProblem):
    """
    The Capacitated Waste Collection Vehicle Routing Problem (CWCVRP).
    """

    NAME = "cwcvrp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the total cost for a set of tours based on capacity constraints.
        """
        CWCVRP.validate_tours(pi)

        if pi.size(-1) == 1:
            overflows = torch.sum(dataset["waste"] >= dataset["max_waste"][:, None], dim=-1)
            # print(f"DEBUG: CWCVRP.get_costs cw_dict={cw_dict}")
            cost = overflows if cw_dict is None else cw_dict["overflows"] * overflows
            c_dict = {
                "overflows": overflows,
                "length": torch.zeros_like(cost),
                "waste": torch.zeros_like(cost),
                "total": cost,
            }
            return cost, c_dict, None

        waste_with_depot = torch.cat(
            (
                torch.full_like(dataset["waste"][:, :1], -VEHICLE_CAPACITY),
                dataset["waste"],
            ),
            1,
        )
        d = waste_with_depot.gather(1, pi).clamp(max=dataset["max_waste"][:, None])
        used_cap = torch.zeros_like(dataset["waste"][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]
            used_cap[used_cap < 0] = 0
            assert (used_cap <= VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        overflow_mask = waste_with_depot >= dataset["max_waste"][:, None]
        visited_mask = torch.zeros_like(waste_with_depot, dtype=torch.bool)
        visited_mask.scatter_(1, pi, True)
        overflows = torch.sum(overflow_mask[:, 1:] & ~visited_mask[:, 1:], dim=-1)

        length = CWCVRP.get_tour_length(dataset, pi, dist_matrix)
        waste = d.clamp(min=0).sum(dim=-1)

        cost = (
            overflows + length - waste
            if cw_dict is None
            else cw_dict["overflows"] * overflows + cw_dict["length"] * length - cw_dict["waste"] * waste
        )
        c_dict = {
            "overflows": overflows,
            "length": length,
            "waste": waste,
            "total": cost,
        }
        return cost, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """Creates a CWCVRP dataset."""
        return WCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """Initializes the CWCVRP state."""
        profit_vars = kwargs.pop("profit_vars", None)
        vehicle_capacity = (
            profit_vars.get("vehicle_capacity", VEHICLE_CAPACITY) if profit_vars is not None else VEHICLE_CAPACITY
        )
        return StateCWCVRP.initialize(*args, vehicle_capacity=vehicle_capacity, **kwargs)


class SDWCVRP(BaseProblem):
    """
    The Split Delivery Waste Collection Vehicle Routing Problem (SDWCVRP).
    """

    NAME = "sdwcvrp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the total cost for split delivery scenarios.
        """
        SDWCVRP.validate_tours(pi)
        batch_size = dataset["waste"].size(0)

        if pi.size(-1) == 1:
            overflow_mask = dataset["waste"] >= dataset["max_waste"][:, None]
            overflows = torch.sum(overflow_mask, dim=-1)
            cost = overflows if cw_dict is None else cw_dict["overflows"] * overflows
            c_dict = {
                "overflows": overflows,
                "length": torch.zeros_like(cost),
                "waste": torch.zeros_like(cost),
                "total": cost,
            }
            return cost, c_dict, None

        wastes = torch.cat(
            (
                torch.full_like(dataset["waste"][:, :1], -VEHICLE_CAPACITY),
                dataset["waste"],
            ),
            1,
        )
        rng = torch.arange(batch_size, device=wastes.device)
        used_cap = torch.zeros_like(dataset["waste"][:, 0])
        for a in pi.transpose(0, 1):
            d = torch.min(wastes[rng, a], VEHICLE_CAPACITY - used_cap)
            wastes[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0

        overflows = torch.sum(wastes[:, 1:] >= dataset["max_waste"][:, None], dim=-1)
        length = SDWCVRP.get_tour_length(dataset, pi, dist_matrix)
        waste = dataset["waste"].sum(dim=-1) - wastes[:, 1:].sum(dim=-1)

        cost = (
            overflows + length - waste
            if cw_dict is None
            else cw_dict["overflows"] * overflows + cw_dict["length"] * length - cw_dict["waste"] * waste
        )
        c_dict = {
            "overflows": overflows,
            "length": length,
            "waste": waste,
            "total": cost,
        }
        return cost, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """Creates an SDWCVRP dataset."""
        return WCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """Initializes the SDWCVRP state."""
        profit_vars = kwargs.pop("profit_vars", None)
        vehicle_capacity = (
            profit_vars.get("vehicle_capacity", VEHICLE_CAPACITY) if profit_vars is not None else VEHICLE_CAPACITY
        )
        return StateSDWCVRP.initialize(*args, vehicle_capacity=vehicle_capacity, **kwargs)

    @classmethod
    def beam_search(cls, *args, **kwargs):
        """Performs beam search for SDWCVRP."""
        assert not kwargs.get("compress_mask", False), "SDWCVRP does not support compression of the mask"
        return super().beam_search(*args, **kwargs)


def make_instance(edge_threshold, edge_strategy, args):
    """Creates a problem instance from raw data."""
    return make_instance_generic(args, edge_threshold, edge_strategy)


def generate_instance(size, edge_threshold, edge_strategy, distribution, bins, *args, graph=None):
    """Generates a random problem instance."""
    if graph is not None:
        depot, loc = graph
    else:
        loc = torch.FloatTensor(size, 2).uniform_(0, 1)
        depot = torch.FloatTensor(2).uniform_(0, 1)

    waste = torch.from_numpy(generate_waste_prize(size, distribution, (depot, loc), 1, bins)).float()
    ret_dict = {
        "loc": loc,
        "depot": depot,
        "waste": waste,
        "max_waste": torch.tensor(MAX_WASTE),
    }
    edges = calculate_edges(loc, edge_threshold, edge_strategy)
    if edges is not None:
        ret_dict["edges"] = edges
    return ret_dict


class WCVRPDataset(BaseDataset):
    """
    Dataset for the Waste Collection Vehicle Routing Problem.
    """

    def __init__(
        self,
        filename=None,
        size=50,
        num_samples=1000000,
        offset=0,
        distribution="unif",
        area="riomaior",
        vertex_strat="mmn",
        number_edges=0,
        edge_strat=None,
        focus_graph=None,
        focus_size=0,
        dist_strat=None,
        waste_type=None,
        dist_matrix_path=None,
    ):
        """Initializes the WCVRP dataset."""
        super(WCVRPDataset, self).__init__()
        num_edges = (
            float(number_edges)
            if isinstance(number_edges, str) and "." in number_edges
            else (int(number_edges) if number_edges is not None else 0)
        )

        if focus_graph is not None and focus_size > 0:
            focus_path = os.path.join(os.getcwd(), "data", "wsr_simulator", "bins_selection", focus_graph)
            tmp_coords, idx, _, _ = load_focus_coords(size, None, area, waste_type, focus_path, focus_size=1)
            dist_matrix = compute_distance_matrix(tmp_coords, dist_strat, dm_filepath=dist_matrix_path, focus_idx=idx)
            depot, loc, _, _ = load_focus_coords(size, vertex_strat, area, waste_type, focus_path, focus_size)
            graph = (torch.from_numpy(depot).float(), torch.from_numpy(loc).float())

            if num_edges > 0 and edge_strat is not None:
                dist_matrix_edges, _, adj_matrix = apply_edges(dist_matrix, num_edges, edge_strat)
                self.edges = torch.from_numpy(adj_matrix)
            else:
                dist_matrix_edges = dist_matrix
                self.edges = None

            self.dist_matrix = torch.from_numpy(dist_matrix_edges).float() / 100
            bins = (
                Bins(
                    size,
                    os.path.join(os.getcwd(), "data", "wsr_simulator"),
                    distribution,
                    area=area,
                    indices=idx[0],
                    waste_type=waste_type,
                )
                if distribution in ["gamma", "emp"]
                else None
            )
        else:
            bins = graph = self.edges = self.dist_matrix = None

        if filename is not None:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = [
                    make_instance(num_edges, edge_strat, args) for args in tqdm(data[offset : offset + num_samples])
                ]
        else:
            self.data = [
                generate_instance(
                    size,
                    num_edges,
                    edge_strat,
                    distribution,
                    bins,
                    graph=(graph[0][i, :], graph[1][i, :, :]) if graph and i < focus_size else None,
                )
                for i in tqdm(range(num_samples))
            ]

        self.size = len(self.data)
