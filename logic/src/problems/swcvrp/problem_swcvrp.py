"""
Stochastic Waste Collection Vehicle Routing Problem (SWCVRP).
"""

import os
import pickle

import torch
from tqdm import tqdm

from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.network import apply_edges, compute_distance_matrix
from logic.src.utils.data_utils import generate_waste_prize, load_focus_coords
from logic.src.utils.definitions import MAX_WASTE, VEHICLE_CAPACITY
from logic.src.utils.problem_utils import calculate_edges

from ..base import BaseDataset, BaseProblem
from .state_scwcvrp import StateSCWCVRP


class SCWCVRP(BaseProblem):
    """
    The Stochastic Capacitated Waste Collection Vehicle Routing Problem.
    """

    NAME = "scwvrp"
    VEHICLE_CAPACITY = 1.0

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the total cost for stochastic scenarios based on real waste levels.
        """
        SCWCVRP.validate_tours(pi)
        batch_size = dataset["real_waste"].size(0)

        if pi.size(-1) == 1:
            overflows = torch.sum(dataset["real_waste"] >= dataset["max_waste"][:, None], dim=-1)
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
                torch.full_like(dataset["real_waste"][:, :1], -VEHICLE_CAPACITY),
                dataset["real_waste"],
            ),
            1,
        )
        rng = torch.arange(batch_size, device=wastes.device)
        used_cap = torch.zeros_like(dataset["real_waste"][:, 0])
        current_node_wastes = wastes.clone()

        for a in pi.transpose(0, 1):
            remaining_cap = VEHICLE_CAPACITY - used_cap
            d = current_node_wastes[rng, a]

            actual_collected = d.clone()
            is_node = a != 0
            violation_mask = is_node & (d > remaining_cap)
            actual_collected[violation_mask] = remaining_cap[violation_mask]

            current_node_wastes[rng, a] -= actual_collected
            used_cap += actual_collected
            used_cap[~is_node] = 0

        overflows = torch.sum(current_node_wastes[:, 1:] >= dataset["max_waste"][:, None], dim=-1)
        length = SCWCVRP.get_tour_length(dataset, pi, dist_matrix)
        waste = dataset["real_waste"].sum(dim=-1) - current_node_wastes[:, 1:].sum(dim=-1)

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
        """Creates an SCWCVRP dataset."""
        return SWCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """Initializes the SCWCVRP state."""
        return StateSCWCVRP.initialize(*args, **kwargs)


def generate_instance(
    size,
    edge_threshold,
    edge_strategy,
    distribution,
    bins,
    *args,
    graph=None,
    noise_mean=0.0,
    noise_variance=0.0,
):
    """Generates a random problem instance with stochastic waste levels."""
    if graph is not None:
        depot, loc = graph
    else:
        loc = torch.FloatTensor(size, 2).uniform_(0, 1)
        depot = torch.FloatTensor(2).uniform_(0, 1)

    real_waste = torch.from_numpy(generate_waste_prize(size, distribution, (depot, loc), 1, bins)).float()

    if noise_variance > 0:
        noise = torch.normal(mean=noise_mean, std=noise_variance**0.5, size=real_waste.size())
        noisy_waste = (real_waste + noise).clamp(min=0.0, max=MAX_WASTE)
    else:
        noisy_waste = real_waste

    ret_dict = {
        "loc": loc,
        "depot": depot,
        "real_waste": real_waste,
        "noisy_waste": noisy_waste,
        "max_waste": torch.tensor(MAX_WASTE),
    }
    edges = calculate_edges(loc, edge_threshold, edge_strategy)
    if edges is not None:
        ret_dict["edges"] = edges
    return ret_dict


class SWCVRPDataset(BaseDataset):
    """
    Dataset for the Stochastic Capacitated Waste Collection Vehicle Routing Problem.
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
        noise_mean=0.0,
        noise_variance=0.0,
    ):
        """Initializes the SWCVRP dataset."""
        super(SWCVRPDataset, self).__init__()
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
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
                self.edges = None
            self.dist_matrix = (
                torch.from_numpy(dist_matrix if "dist_matrix_edges" not in locals() else dist_matrix_edges).float()
                / 100
            )
            bins = (
                Bins(
                    size,
                    os.path.join(os.getcwd(), "data", "wsr_simulator"),
                    distribution,
                    area=area,
                    indices=idx[0],
                    waste_type=waste_type,
                    noise_mean=self.noise_mean,
                    noise_variance=self.noise_variance,
                )
                if distribution in ["gamma", "emp"]
                else None
            )
        else:
            bins = graph = self.edges = self.dist_matrix = None

        if filename is not None:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = []
                for item in tqdm(data[offset : offset + num_samples]):
                    depot, loc, real_waste, noisy_waste, max_waste = item
                    real_waste_t, noisy_waste_t = (
                        torch.FloatTensor(real_waste),
                        torch.FloatTensor(noisy_waste),
                    )
                    if real_waste_t.dim() > 1:
                        real_waste_t = real_waste_t[0]
                    if noisy_waste_t.dim() > 1:
                        noisy_waste_t = noisy_waste_t[0]
                    instance = {
                        "depot": torch.FloatTensor(depot),
                        "loc": torch.FloatTensor(loc),
                        "real_waste": real_waste_t,
                        "noisy_waste": noisy_waste_t,
                        "max_waste": torch.FloatTensor(max_waste),
                    }
                    edges = calculate_edges(instance["loc"], num_edges, edge_strat)
                    if edges is not None:
                        instance["edges"] = edges
                    self.data.append(instance)
        else:
            self.data = [
                generate_instance(
                    size,
                    num_edges,
                    edge_strat,
                    distribution,
                    bins,
                    graph=(graph[0][i, :], graph[1][i, :, :]) if graph and i < focus_size else None,
                    noise_mean=self.noise_mean,
                    noise_variance=self.noise_variance,
                )
                for i in tqdm(range(num_samples))
            ]

        self.size = len(self.data)
