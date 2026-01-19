"""
Vehicle Routing Problem with Profits (VRPP) and its variants.
"""

import os
import pickle

import torch
from tqdm import tqdm

from logic.src.pipeline.simulations.bins import Bins
from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
from logic.src.pipeline.simulations.network import apply_edges, compute_distance_matrix
from logic.src.utils.data_utils import generate_waste_prize, load_focus_coords
from logic.src.utils.definitions import MAX_WASTE
from logic.src.utils.task_utils import calculate_edges, make_instance_generic

from ..base import BaseDataset, BaseProblem
from .state_cvrpp import StateCVRPP
from .state_vrpp import StateVRPP

# Default values for profit/cost parameters (can be overridden by VRPPDataset.__init__)
COST_KM = 1.0  # Cost per km traveled
REVENUE_KG = 0.1625  # Revenue per kg collected
BIN_CAPACITY = 100.0  # Bin capacity in kg
VEHICLE_CAPACITY = 100.0  # Vehicle capacity (normalized)


class VRPP(BaseProblem):
    """
    The Vehicle Routing Problem with Profits (VRPP).
    """

    NAME = "vrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the negative profit (cost) for a set of tours.
        """
        VRPP.validate_tours(pi)

        if pi.size(-1) == 1:
            profit = torch.zeros_like(dataset["max_waste"]).to(pi.device)
            c_dict = {"length": profit, "waste": profit, "total": profit}
            return profit, c_dict, None

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        w = waste_with_depot.gather(1, pi).clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)

        length = VRPP.get_tour_length(dataset, pi, dist_matrix)

        negative_profit = (
            length * COST_KM - waste * REVENUE_KG
            if cw_dict is None
            else cw_dict["length"] * length * COST_KM - cw_dict["waste"] * waste * REVENUE_KG
        )
        c_dict = {"length": length, "waste": waste, "total": negative_profit}
        return negative_profit, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """Creates a VRPP dataset."""
        return VRPPDataset(*args, **kwargs)

    @staticmethod
    def make_state(input, edges=None, cost_weights=None, dist_matrix=None, *args, **kwargs):
        """Initializes the VRPP state."""
        if "profit_vars" not in kwargs or kwargs["profit_vars"] is None:
            kwargs["profit_vars"] = {
                "cost_km": COST_KM,
                "revenue_kg": REVENUE_KG,
            }
        return StateVRPP.initialize(
            input,
            edges,
            cost_weights=cost_weights,
            dist_matrix=dist_matrix,
            *args,
            **kwargs,
        )


class CVRPP(BaseProblem):
    """
    The Capacitated Vehicle Routing Problem with Profits (CVRPP).
    """

    NAME = "cvrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the negative profit (cost) for capacitated tours.
        """
        CVRPP.validate_tours(pi)

        if pi.size(-1) == 1:
            profit = torch.zeros_like(dataset["max_waste"]).to(pi.device)
            c_dict = {"length": profit, "waste": profit, "total": profit}
            return profit, c_dict, None

        waste_with_depot_reset = torch.cat(
            (
                torch.full_like(dataset["waste"][:, :1], -VEHICLE_CAPACITY),
                dataset["waste"],
            ),
            1,
        )
        d_reset = waste_with_depot_reset.gather(1, pi)
        used_cap = torch.zeros_like(dataset["waste"][:, 0])
        for i in range(pi.size(1)):
            used_cap += d_reset[:, i]
            used_cap[used_cap < 0] = 0
            assert (used_cap <= VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        w = waste_with_depot.gather(1, pi).clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)

        length = CVRPP.get_tour_length(dataset, pi, dist_matrix)

        negative_profit = (
            length * COST_KM - waste * REVENUE_KG
            if cw_dict is None
            else cw_dict["length"] * length * COST_KM - cw_dict["waste"] * waste * REVENUE_KG
        )
        c_dict = {"length": length, "waste": waste, "total": negative_profit}
        return negative_profit, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """Creates a CVRPP dataset."""
        return VRPPDataset(*args, **kwargs)

    @staticmethod
    def make_state(input, edges=None, cost_weights=None, dist_matrix=None, *args, **kwargs):
        """Initializes the CVRPP state."""
        if "profit_vars" not in kwargs or kwargs["profit_vars"] is None:
            kwargs["profit_vars"] = {
                "cost_km": COST_KM,
                "revenue_kg": REVENUE_KG,
                "bin_capacity": BIN_CAPACITY,
                "vehicle_capacity": VEHICLE_CAPACITY,
            }
        return StateCVRPP.initialize(
            input,
            edges,
            cost_weights=cost_weights,
            dist_matrix=dist_matrix,
            *args,
            **kwargs,
        )


def make_instance(edge_threshold, edge_strategy, args):
    """Creates a problem instance from raw data."""
    return make_instance_generic(args, edge_threshold, edge_strategy)


def generate_instance(size, edge_threshold, edge_strategy, distribution, bins, graph=None):
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


class VRPPDataset(BaseDataset):
    """
    Dataset for the Vehicle Routing Problem with Profits.
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
        """Initializes the VRPP dataset."""
        super(VRPPDataset, self).__init__()
        num_edges = (
            float(number_edges)
            if isinstance(number_edges, str) and "." in number_edges
            else (int(number_edges) if number_edges is not None else 0)
        )

        global COST_KM, REVENUE_KG, BIN_CAPACITY, VEHICLE_CAPACITY
        (
            VEHICLE_CAPACITY,
            REVENUE_KG,
            DENSITY,
            COST_KM,
            VOLUME,
        ) = load_area_and_waste_type_params(area, waste_type)
        BIN_CAPACITY = VOLUME * DENSITY
        VEHICLE_CAPACITY = VEHICLE_CAPACITY / 100

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
