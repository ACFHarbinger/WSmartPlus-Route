"""
Definition of the Vehicle Routing Problem with Profits (VRPP) and its variants.

This module defines the problem logic, cost calculations, and dataset handling
for the profit-maximizing routing problem.
"""

import os
import torch
import pickle

from logic.src.utils.definitions import MAX_WASTE
from tqdm import tqdm
from torch.utils.data import Dataset
from .state_vrpp import StateVRPP
from .state_cvrpp import StateCVRPP
from scipy.spatial.distance import pdist, squareform
from logic.src.utils.beam_search import beam_search
from logic.src.utils.data_utils import load_focus_coords, generate_waste_prize
from logic.src.utils.graph_utils import get_edge_idx_dist, get_adj_knn, adj_to_idx
from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params
from logic.src.pipeline.simulator.network import compute_distance_matrix, apply_edges


class VRPP(object):
    """
    The Vehicle Routing Problem with Profits (VRPP).

    In this problem, each node has a profit (waste), and the goal is to find a route
    that maximizes (Revenue * Collected Waste - Cost * Distance).
    """

    NAME = "vrpp"  # Vehicle routing problem with profits

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the negative profit (cost) for a set of tours.

        Args:
            dataset (dict): The dataset containing 'waste', 'max_waste', etc.
            pi (Tensor): The sequence of visited nodes (batch_size, tour_length).
            cw_dict (dict): Weights for different cost components (e.g., length, waste).
            dist_matrix (Tensor, optional): Precomputed distance matrix.

        Returns:
            tuple: (negative_profit, cost_breakdown_dict, None)
        """
        if (
            pi.size(-1) == 1
        ):  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            profit = torch.zeros_like(dataset["max_waste"]).to(pi.device)
            c_dict = {"length": profit, "waste": profit, "total": profit}
            return profit, c_dict, None

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])
        ).all(), "Duplicates"
        waste_with_depot = torch.cat(
            (torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1
        )
        w = waste_with_depot.gather(1, pi).clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)
        if dist_matrix is not None:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst = torch.max(
                dst_mask
                * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
                dim=1,
            ).indices
            length = (
                dist_matrix[
                    0,
                    dst_vertices[
                        torch.arange(dst_vertices.size(0), device=dst_vertices.device),
                        last_dst,
                    ],
                    0,
                ]
                + dists.sum(dim=1)
                + dist_matrix[0, 0, src_vertices[:, 0]]
            )
        else:
            # Gather dataset in order of tour
            loc_with_depot = torch.cat((dataset["depot"][:, None, :], dataset["loc"]), 1)
            d = loc_with_depot.gather(
                1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1))
            )
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
                + (d[:, 0] - dataset["depot"]).norm(p=2, dim=-1)  # Depot to first
                + (d[:, -1] - dataset["depot"]).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
            )

        negative_profit = (
            length * COST_KM - waste * REVENUE_KG
            if cw_dict is None
            else cw_dict["length"] * length * COST_KM
            - cw_dict["waste"] * waste * REVENUE_KG
        )
        c_dict = {"length": length, "waste": waste, "total": negative_profit}
        return negative_profit, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """
        Creates an instance of the VRPPDataset.

        Returns:
            VRPPDataset: The initialized dataset.
        """
        return VRPPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """
        Creates an initial state for the VRPP.

        Returns:
            StateVRPP: The initial state.
        """
        if "profit_vars" not in kwargs or kwargs["profit_vars"] is None:
            kwargs["profit_vars"] = {
                "cost_km": COST_KM,
                "revenue_kg": REVENUE_KG,
                "bin_capacity": BIN_CAPACITY,
            }
        return StateVRPP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(
        input,
        beam_size,
        cost_weights,
        edges=None,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
    ):
        """
        Performs beam search to find high-quality solutions for VRPP.

        Args:
            input (dict): The input data instance.
            beam_size (int): The number of beams to maintain.
            cost_weights (dict): Weights for the cost function.
            edges (Tensor, optional): Edge information if using graph models.
            expand_size (int, optional): Number of expansions per beam.
            compress_mask (bool): Whether to compress the mask for efficiency.
            model (nn.Module, optional): The model used for node selection.
            max_calc_batch_size (int): Maximum batch size for calculations.

        Returns:
            Tensor: The best tours found.
        """
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = VRPP.make_state(
            input,
            edges,
            cost_weights,
            visited_dtype=torch.int64 if compress_mask else torch.uint8,
        )
        return beam_search(state, beam_size, propose_expansions)


class CVRPP(object):
    """
    The Capacitated Vehicle Routing Problem with Profits (CVRPP).

    Similar to VRPP, but includes vehicle capacity constraints.
    """

    NAME = "cvrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Calculates the negative profit (cost) for a set of tours, validating capacity.

        Args:
            dataset (dict): The dataset containing 'waste', 'max_waste', etc.
            pi (Tensor): The sequence of visited nodes (batch_size, tour_length).
            cw_dict (dict): Weights for different cost components.
            dist_matrix (Tensor, optional): Precomputed distance matrix.

        Returns:
            tuple: (negative_profit, cost_breakdown_dict, None)
        """
        if pi.size(-1) == 1:
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            profit = torch.zeros_like(dataset["max_waste"]).to(pi.device)
            c_dict = {"length": profit, "waste": profit, "total": profit}
            return profit, c_dict, None

        # Check for duplicates (except depot)
        sorted_pi = pi.data.sort(1)[0]
        assert (
            (sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])
        ).all(), "Duplicates"

        # --- Capacity Validation Logic (Similar to CVRP) ---
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
            # Reset at depot (if used_cap becomes negative due to -VEHICLE_CAPACITY, clamp to 0)
            used_cap[used_cap < 0] = 0
            # Check capacity constraint with floating point tolerance
            if (used_cap > VEHICLE_CAPACITY + 1e-5).any():
                assert (
                    used_cap <= VEHICLE_CAPACITY + 1e-5
                ).all(), "Used more than capacity"

        # --- Profit Calculation (Similar to VRPP) ---
        # For profit, we sum actual waste collected (depot has 0)
        waste_with_depot = torch.cat(
            (torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1
        )
        # Clamp waste to max_waste (per bin)
        w = waste_with_depot.gather(1, pi).clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)
        if dist_matrix is not None:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst = torch.max(
                dst_mask
                * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
                dim=1,
            ).indices
            length = (
                dist_matrix[
                    0,
                    dst_vertices[
                        torch.arange(dst_vertices.size(0), device=dst_vertices.device),
                        last_dst,
                    ],
                    0,
                ]
                + dists.sum(dim=1)
                + dist_matrix[0, 0, src_vertices[:, 0]]
            )
        else:
            # Gather dataset in order of tour
            loc_with_depot = torch.cat((dataset["depot"][:, None, :], dataset["loc"]), 1)
            d = loc_with_depot.gather(
                1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1))
            )
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)
                + (d[:, 0] - dataset["depot"]).norm(p=2, dim=-1)
                + (d[:, -1] - dataset["depot"]).norm(p=2, dim=-1)
            )

        negative_profit = (
            length * COST_KM - waste * REVENUE_KG
            if cw_dict is None
            else cw_dict["length"] * length * COST_KM
            - cw_dict["waste"] * waste * REVENUE_KG
        )
        c_dict = {"length": length, "waste": waste, "total": negative_profit}
        return negative_profit, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """
        Creates an instance of the VRPPDataset.

        Returns:
            VRPPDataset: The initialized dataset.
        """
        return VRPPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """
        Creates an initial state for the CVRPP.

        Returns:
            StateCVRPP: The initial state.
        """
        if "profit_vars" not in kwargs or kwargs["profit_vars"] is None:
            kwargs["profit_vars"] = {
                "cost_km": COST_KM,
                "revenue_kg": REVENUE_KG,
                "bin_capacity": BIN_CAPACITY,
                "vehicle_capacity": VEHICLE_CAPACITY,
            }
        return StateCVRPP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(
        input,
        beam_size,
        cost_weights,
        edges=None,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
    ):
        """
        Performs beam search to find high-quality solutions for CVRPP.

        Args:
            input (dict): The input data instance.
            beam_size (int): The number of beams to maintain.
            cost_weights (dict): Weights for the cost function.
            edges (Tensor, optional): Edge information if using graph models.
            expand_size (int, optional): Number of expansions per beam.
            compress_mask (bool): Whether to compress the mask for efficiency.
            model (nn.Module, optional): The model used for node selection.
            max_calc_batch_size (int): Maximum batch size for calculations.

        Returns:
            Tensor: The best tours found.
        """
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = CVRPP.make_state(
            input,
            edges,
            cost_weights,
            visited_dtype=torch.int64 if compress_mask else torch.uint8,
        )
        return beam_search(state, beam_size, propose_expansions)


def make_instance(edge_threshold, edge_strategy, args):
    """
    Creates a single VRPP instance from raw data.

    Args:
        edge_threshold (float): Threshold for edge generation.
        edge_strategy (str): Strategy for edge generation ('dist' or 'knn').
        args (tuple): Raw data (depot, loc, waste, max_waste, ...).

    Returns:
        dict: A dictionary containing instance data.
    """
    depot, loc, waste, max_waste, *args = args
    ret_dict = {
        "loc": torch.FloatTensor(loc),
        "depot": torch.FloatTensor(depot),
        "waste": torch.tensor(waste, dtype=torch.float),
        "max_waste": torch.tensor(max_waste, dtype=torch.float),
    }
    if ret_dict["waste"].size(dim=0) > 1:
        for day_id in range(1, len(waste)):
            ret_dict["fill{}".format(day_id)] = ret_dict["waste"][day_id]

    if len(ret_dict["waste"].size()) > 1:
        ret_dict["waste"] = ret_dict["waste"][0]

    if edge_threshold > 0:
        distance_matrix = squareform(pdist(ret_dict["loc"], metric="euclidean"))
        if edge_strategy == "dist":
            ret_dict["edges"] = torch.tensor(
                get_edge_idx_dist(distance_matrix, edge_threshold)
            ).to(dtype=torch.long)
        else:
            assert edge_strategy == "knn"
            neg_adj_matrix = get_adj_knn(distance_matrix, edge_threshold)
            ret_dict["edges"] = torch.tensor(adj_to_idx(neg_adj_matrix)).to(
                dtype=torch.long
            )
    return ret_dict


def generate_instance(
    size, edge_threshold, edge_strategy, distribution, bins, graph=None
):
    """
    Gathers or generates a single random VRPP instance.

    Args:
        size (int): Number of nodes.
        edge_threshold (float): Threshold for edge generation.
        edge_strategy (str): Strategy for edge generation.
        distribution (str): Waste distribution.
        bins (Bins): Bins object for simulation.
        graph (tuple, optional): Preset coordinates (depot, loc).

    Returns:
        dict: A dictionary containing instance data.
    """
    if graph is not None:
        depot, loc = graph
    else:
        loc = torch.FloatTensor(size, 2).uniform_(0, 1)
        depot = torch.FloatTensor(2).uniform_(0, 1)

    waste = torch.from_numpy(
        generate_waste_prize(size, distribution, (depot, loc), 1, bins)
    ).float()
    ret_dict = {
        "loc": loc,
        "depot": depot,
        "waste": waste,
        "max_waste": torch.tensor(MAX_WASTE),
    }
    if edge_threshold > 0:
        distance_matrix = squareform(pdist(ret_dict["loc"], metric="euclidean"))
        if edge_strategy == "dist":
            ret_dict["edges"] = torch.tensor(
                get_edge_idx_dist(distance_matrix, edge_threshold)
            ).to(dtype=torch.long)
        else:
            assert edge_strategy == "knn"
            neg_adj_matrix = get_adj_knn(distance_matrix, edge_threshold)
            ret_dict["edges"] = torch.tensor(adj_to_idx(neg_adj_matrix)).to(
                dtype=torch.long
            )
    return ret_dict


class VRPPDataset(Dataset):
    """
    Dataset for the Vehicle Routing Problem with Profits.

    Supports loading from pickle files or generating instances on the fly.
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
        """
        Initializes the VRPP dataset.

        Args:
            filename (str, optional): Path to a .pkl file to load data from.
            size (int): Graph size.
            num_samples (int): Number of instances.
            offset (int): Loading offset.
            distribution (str): Waste distribution.
            area (str): Geographic area.
            vertex_strat (str): Vertex placement strategy.
            number_edges (int): Edge count or threshold.
            edge_strat (str): Edge strategy.
            focus_graph (str, optional): Fixed graph to focus on.
            focus_size (int): Number of focus instances.
            dist_strat (str, optional): Distance strategy.
            waste_type (str, optional): Waste type.
            dist_matrix_path (str, optional): Path to dist matrix.
        """
        super(VRPPDataset, self).__init__()
        dist = distribution
        self.data_set = []
        if isinstance(number_edges, str):
            if "." in number_edges:
                num_edges = float(number_edges)
            else:
                num_edges = int(number_edges)
        else:
            num_edges = number_edges if number_edges is not None else 0

        global COST_KM
        global REVENUE_KG
        global BIN_CAPACITY
        global VEHICLE_CAPACITY
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
            focus_path = os.path.join(
                os.getcwd(), "data", "wsr_simulator", "bins_selection", focus_graph
            )
            tmp_coords, idx, _, _ = load_focus_coords(
                size, None, area, waste_type, focus_path, focus_size=1
            )
            dist_matrix = compute_distance_matrix(
                tmp_coords, dist_strat, dm_filepath=dist_matrix_path, focus_idx=idx
            )
            depot, loc, _, _ = load_focus_coords(
                size, vertex_strat, area, waste_type, focus_path, focus_size
            )
            graph = (torch.from_numpy(depot).float(), torch.from_numpy(loc).float())
            if num_edges > 0 and edge_strat is not None:
                dist_matrix_edges, _, adj_matrix = apply_edges(
                    dist_matrix, num_edges, edge_strat
                )
                # adj_matrix = get_adj_knn(dist_matrix[1:, 1:], num_edges, negative=False)
                self.edges = torch.from_numpy(adj_matrix)
                # self.edges = torch.tensor(adj_to_idx(neg_adj_matrix)).to(dtype=torch.long)
                # if edge_strat is None: num_edges = 0
            else:
                dist_matrix_edges = dist_matrix
                self.edges = None
            self.dist_matrix = torch.from_numpy(dist_matrix_edges).float() / 100
            if distribution in ["gamma", "emp"]:
                bins = Bins(
                    size,
                    os.path.join(os.getcwd(), "data", "wsr_simulator"),
                    distribution,
                    area=area,
                    indices=idx[0],
                    waste_type=waste_type,
                )
            else:
                bins = None
        else:
            idx = None
            bins = None
            graph = None
            focus_path = None
            self.edges = None
            self.dist_matrix = None

        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"
            print(f"Loading data from {filename}...")
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = [
                    make_instance(num_edges, edge_strat, args)
                    for args in tqdm(data[offset : offset + num_samples])
                ]
        else:
            print("Generating data...")
            self.data = [
                generate_instance(size, num_edges, edge_strat, dist, bins)
                if i >= focus_size
                else generate_instance(
                    size,
                    num_edges,
                    edge_strat,
                    dist,
                    bins,
                    graph=(graph[0][i, :], graph[1][i, :, :]),
                )
                for i in tqdm(range(num_samples))
            ]
        self.size = len(self.data)

    def __len__(self):
        """Returns the number of instances in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Returns the instance at the given index."""
        return self.data[idx]

    def __setitem__(self, key, values):
        """Updates instances in the dataset."""

        def __update_item(inst, key, value):
            inst[key] = value
            return inst

        self.data = [__update_item(x, key, val) for x, val in zip(self.data, values)]