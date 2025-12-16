import os
import torch
import pickle

from logic.src.utils.definitions import MAX_WASTE
from tqdm import tqdm
from torch.utils.data import Dataset
from .state_wcrp import StateWCRP
from .state_cwcvrp import StateCWCVRP
from .state_sdwcvrp import StateSDWCVRP
from logic.src.utils.beam_search import beam_search
from scipy.spatial.distance import pdist, squareform
from logic.src.utils.data_utils import load_focus_coords, generate_waste_prize
from logic.src.utils.graph_utils import get_edge_idx_dist, get_adj_knn, adj_to_idx
from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.network import compute_distance_matrix, apply_edges


class WCRP(object):
    NAME = 'wcrp'  # Waste collection routing problem

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            overflow_mask = dataset['waste'] >= dataset['max_waste'][:, None]
            overflows = torch.sum(overflow_mask, dim=-1, dtype=torch.float)
            #excess_waste = (dataset['waste'] - dataset['max_waste'][:, None]).clamp(min=0)
            #waste_lost = torch.sum(excess_waste, dim=-1, dtype=torch.float)
            cost = overflows if cw_dict is None \
            else cw_dict['overflows'] * overflows
            c_dict = {'overflows': overflows, 'length': torch.zeros_like(overflows).to(pi.device),
                    'waste': torch.zeros_like(overflows).to(pi.device), 'total': cost}
            return cost, c_dict, None

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"
        waste_with_depot = torch.cat(
            (
                torch.zeros_like(dataset['waste'][:, :1]),
                dataset['waste']
            ),
            1
        )
        w = waste_with_depot.gather(1, pi).clamp(max=dataset['max_waste'][:, None])
        waste = w.sum(dim=-1)

        # Get masks for bins with overflows and present in tour
        overflow_mask = waste_with_depot >= dataset['max_waste'][:, None]
        #batch_dim, node_dim = overflow_mask.size()
        #visited_mask = torch.zeros((batch_dim, node_dim), dtype=torch.bool).to(sorted_pi.device)
        #col_idx = sorted_pi[sorted_pi != 0]
        #row_idx = torch.arange(batch_dim, device=sorted_pi.device).repeat_interleave((sorted_pi != 0).sum(dim=1))
        #visited_mask[row_idx, col_idx] = True

        # Compute number of overflows
        #overflows = torch.sum(overflow_mask & ~visited_mask, dim=-1)
        overflows = torch.sum(overflow_mask, dim=-1)

        # Compute the amount of waste above max_waste
        #excess_waste = (waste_with_depot - dataset['max_waste'][:, None]).clamp(min=0)
        #waste_lost = torch.sum(excess_waste, dim=-1)

        if dist_matrix is not None:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst = torch.max(dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device), dim=1).indices
            length = dist_matrix[0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0] + dists.sum(dim=1) + dist_matrix[0, 0, src_vertices[:, 0]]
        else:
            # Gather dataset in order of tour
            loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
                + (d[:, 0] - dataset['depot']).norm(p=2, dim=-1)  # Depot to first
                + (d[:, -1] - dataset['depot']).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
            )
        
        cost = overflows + length - waste if cw_dict is None \
        else cw_dict['overflows'] * overflows + cw_dict['length'] * length - cw_dict['waste'] * waste
        c_dict = {'overflows': overflows, 'length': length, 'waste': waste, 'total': cost}
        return cost, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return WCRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        kwargs.pop('profit_vars', None)
        return StateWCRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, cost_weights, edges=None, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size)

        state = WCRP.make_state(input, edges, cost_weights, visited_dtype=torch.int64 if compress_mask else torch.uint8)
        return beam_search(state, beam_size, propose_expansions)


class CWCVRP(object):
    NAME = 'cwcvrp'  # Capacitated Waste Collection Vehicle Routing Problem
    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        batch_size, graph_size = dataset['waste'].size()

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        waste_with_depot = torch.cat(
            (
                torch.full_like(dataset['waste'][:, :1], -CWCVRP.VEHICLE_CAPACITY),
                dataset['waste']
            ),
            1
        )
        d = waste_with_depot.gather(1, pi).clamp(max=dataset['max_waste'][:, None])
        used_cap = torch.zeros_like(dataset['waste'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited

            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CWCVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Get masks for bins with overflows and present in tour
        overflow_mask = waste_with_depot >= dataset['max_waste'][:, None]
        batch_dim, node_dim = overflow_mask.size()
        visited_mask = torch.zeros((batch_dim, node_dim), dtype=torch.bool).to(sorted_pi.device)
        col_idx = sorted_pi[sorted_pi != 0]
        row_idx = torch.arange(batch_dim, device=sorted_pi.device).repeat_interleave((sorted_pi != 0).sum(dim=1))
        visited_mask[row_idx, col_idx] = True

        # Compute number of overflows in unvisited bins
        overflows = torch.sum(overflow_mask & ~visited_mask, dim=-1)

        # Compute the amount of waste above max_waste in unvisited bins
        #excess_waste = (waste_with_depot - dataset['max_waste'][:, None]).clamp(min=0)
        #waste_lost = torch.sum(excess_waste * (~visited_mask), dim=-1)

        if dist_matrix:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst = torch.max(dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device), dim=1).indices
            length = dist_matrix[0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0] + dists.sum(dim=1) + dist_matrix[0, 0, src_vertices[:, 0]]
        else:
            # Gather dataset in order of tour
            loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
                + (d[:, 0] - dataset['depot']).norm(p=2, dim=-1)  # Depot to first
                + (d[:, -1] - dataset['depot']).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
            )

        waste = d.sum(dim=-1)
        cost = overflows + length - waste if cw_dict is None \
        else cw_dict['overflows'] * overflows + cw_dict['length'] * length - cw_dict['waste'] * waste
        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return WCRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCWCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, cost_weights, edges=None, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size)

        state = CWCVRP.make_state(input, edges, cost_weights, visited_dtype=torch.int64 if compress_mask else torch.uint8)
        return beam_search(state, beam_size, propose_expansions)


class SDWCVRP(object):
    NAME = 'sdwcvrp'  # Split Delivery Waste Collection Vehicle Routing Problem
    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        batch_size, graph_size = dataset['waste'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['waste'][:, :1], -SDWCVRP.VEHICLE_CAPACITY),
                dataset['waste']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['waste'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDWCVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Get masks for bins with overflows and present in tour
        sorted_pi = pi.data.sort(1)[0]
        overflow_mask = demands >= dataset['max_waste'][:, None]
        batch_dim, node_dim = overflow_mask.size()
        visited_mask = torch.zeros((batch_dim, node_dim), dtype=torch.bool).to(sorted_pi.device)
        col_idx = sorted_pi[sorted_pi != 0]
        row_idx = torch.arange(batch_dim, device=sorted_pi.device).repeat_interleave((sorted_pi != 0).sum(dim=1))
        visited_mask[row_idx, col_idx] = True

        # Compute number of overflows in unvisited bins
        overflows = torch.sum(overflow_mask & ~visited_mask, dim=-1)

        # Compute the amount of waste above max_waste in unvisited bins
        #excess_waste = (demands - dataset['max_waste'][:, None]).clamp(min=0)
        #waste_lost = torch.sum(excess_waste * (~visited_mask), dim=-1)

        if dist_matrix is not None:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst = torch.max(dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device), dim=1).indices
            length = dist_matrix[0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0] + dists.sum(dim=1) + dist_matrix[0, 0, src_vertices[:, 0]]
        else:
            # Gather dataset in order of tour
            loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

            # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
                + (d[:, 0] - dataset['depot']).norm(p=2, dim=-1)  # Depot to first
                + (d[:, -1] - dataset['depot']).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
            )

        waste = demands.sum(dim=-1)
        cost = overflows + length - waste if cw_dict is None \
        else cw_dict['overflows'] * overflows + cw_dict['length'] * length + cw_dict['waste'] * waste
        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return WCRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDWCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, cost_weights, edges=None, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDWCVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size)

        state = SDWCVRP.make_state(input, edges, cost_weights)
        return beam_search(state, beam_size, propose_expansions)


def make_instance(edge_threshold, edge_strategy, args):
    depot, loc, waste, max_waste, *args = args
    ret_dict = {
        'loc': torch.FloatTensor(loc),
        'depot': torch.FloatTensor(depot),
        'waste': torch.tensor(waste, dtype=torch.float),
        'max_waste': torch.tensor(max_waste, dtype=torch.float)
    }
    if ret_dict['waste'].size(dim=0) > 1:
        for day_id in range(1, len(waste)):
            ret_dict["fill{}".format(day_id)] = ret_dict['waste'][day_id]
    
    if len(ret_dict['waste'].size()) > 1:
        ret_dict['waste'] = ret_dict['waste'][0]

    if edge_threshold > 0:
        distance_matrix = squareform(pdist(ret_dict['loc'], metric='euclidean'))
        if edge_strategy == 'dist':
            ret_dict['edges'] = torch.tensor(get_edge_idx_dist(distance_matrix, edge_threshold)).to(dtype=torch.long)
        else:
            assert edge_strategy == 'knn'
            neg_adj_matrix = get_adj_knn(distance_matrix, edge_threshold)
            ret_dict['edges'] = torch.tensor(adj_to_idx(neg_adj_matrix)).to(dtype=torch.long)
    return ret_dict


def generate_instance(size, edge_threshold, edge_strategy, distribution, bins, *args, graph=None):
    if graph is not None:
        depot, loc = graph
    else:
        loc = torch.FloatTensor(size, 2).uniform_(0, 1)
        depot = torch.FloatTensor(2).uniform_(0, 1)

    waste = torch.from_numpy(generate_waste_prize(size, distribution, (depot, loc), 1, bins)).float()
    ret_dict = {
        'loc': loc,
        'depot': depot,
        'waste': waste,
        'max_waste': torch.tensor(MAX_WASTE)
    }
    if edge_threshold > 0:
        distance_matrix = squareform(pdist(ret_dict['loc'], metric='euclidean'))
        if edge_strategy == 'dist':
            ret_dict['edges'] = torch.tensor(get_edge_idx_dist(distance_matrix, edge_threshold)).to(dtype=torch.long)
        else:
            assert edge_strategy == 'knn'
            neg_adj_matrix = get_adj_knn(distance_matrix, edge_threshold)
            ret_dict['edges'] = torch.tensor(adj_to_idx(neg_adj_matrix)).to(dtype=torch.long)
    return ret_dict


class WCRPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution='unif', area="riomaior", vertex_strat="mmn", 
                number_edges=0, edge_strat=None, focus_graph=None, focus_size=0, dist_strat=None, waste_type=None, dist_matrix_path=None):
        super(WCRPDataset, self).__init__()
        dist = distribution
        self.data_set = []
        if isinstance(number_edges, str):
            if '.' in number_edges:
                num_edges = float(number_edges)
            else:
                num_edges = int(number_edges)
        else:
            num_edges = number_edges if number_edges is not None else 0

        if focus_graph is not None and focus_size > 0:
            focus_path = os.path.join(os.getcwd(), "data", "wsr_simulator", "bins_selection", focus_graph)
            tmp_coords, idx = load_focus_coords(size, None, area, waste_type, focus_path, focus_size=1)
            dist_matrix = compute_distance_matrix(tmp_coords, dist_strat, dm_filepath=dist_matrix_path, focus_idx=idx)
            depot, loc, _, _ = load_focus_coords(size, vertex_strat, area, waste_type, focus_path, focus_size)
            graph = (torch.from_numpy(depot).float(), torch.from_numpy(loc).float())
            if num_edges > 0:
                dist_matrix_edges, _, adj_matrix = apply_edges(dist_matrix, num_edges, edge_strat)
                #adj_matrix = get_adj_knn(dist_matrix[1:, 1:], num_edges, negative=False)
                self.edges = torch.from_numpy(adj_matrix)
                #self.edges = torch.tensor(adj_to_idx(neg_adj_matrix)).to(dtype=torch.long)
                if edge_strat is None: num_edges = 0
            else:
                dist_matrix_edges = dist_matrix
                self.edges = None
            self.dist_matrix = torch.from_numpy(dist_matrix_edges).float() / 100
            if distribution in ['gamma', 'emp']:
                bins = Bins(size, os.path.join(os.getcwd(), "data", "wsr_simulator"), distribution, area=area, indices=idx[0], waste_type=waste_type)
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
            assert os.path.splitext(filename)[1] == '.pkl'
            print(f"Loading data from {filename}...")
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    make_instance(num_edges, edge_strat, args) 
                    for args in tqdm(data[offset:offset+num_samples])
                ]
        else:
            print("Generating data...")
            args = (focus_path, idx, area)
            self.data = [
                generate_instance(size, num_edges, edge_strat, dist, bins, args) if i >= focus_size 
                else generate_instance(size, num_edges, edge_strat, dist, bins, graph=(graph[0][i, :], graph[1][i, :, :]))
                for i in tqdm(range(num_samples))
            ]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, key, values):
        def __update_item(inst, key, value):
            inst[key] = value
            return inst
        self.data = [__update_item(x, key, val) for x, val in zip(self.data, values)]