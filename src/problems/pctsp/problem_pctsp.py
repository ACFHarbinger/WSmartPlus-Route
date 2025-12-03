import os
import torch
import pickle

from src.utils.definitions import MAX_LENGTHS
from tqdm import tqdm
from torch.utils.data import Dataset
from .state_pctsp import StatePCTSP
from src.utils.beam_search import beam_search
from src.utils.data_utils import load_focus_coords
from scipy.spatial.distance import pdist, squareform
from src.utils.graph_utils import get_edge_idx_dist, get_adj_knn, adj_to_idx
from src.pipeline.simulator.network import compute_distance_matrix, apply_edges


class PCTSP(object):
    NAME = 'pctsp'  # Prize Collecting TSP, without depot, with penalties

    @staticmethod
    def _get_costs(dataset, pi, cw_dict, dist_matrix=None, stochastic=False):
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"

        prize = dataset['stochastic_prize'] if stochastic else dataset['deterministic_prize']
        prize_with_depot = torch.cat(
            (
                torch.zeros_like(prize[:, :1]),
                prize
            ),
            1
        )
        p = prize_with_depot.gather(1, pi)

        # Either prize constraint should be satisfied or all prizes should be visited
        assert (
            (p.sum(-1) >= 1 - 1e-5) |
            (sorted_pi.size(-1) - (sorted_pi == 0).int().sum(-1) == dataset['loc'].size(-2))
        ).all(), "Total prize does not satisfy min total prize"
        penalty_with_depot = torch.cat(
            (
                torch.zeros_like(dataset['penalty'][:, :1]),
                dataset['penalty']
            ),
            1
        )
        pen = penalty_with_depot.gather(1, pi)
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
        # We want to maximize total prize but code minimizes so return negative
        # Incurred penalty cost is total penalty cost - saved penalty costs of nodes visited
        cost = length + dataset['penalty'].sum(-1) - pen.sum(-1) if cw_dict is None \
        else cw_dict['length'] * length + cw_dict['penalty'] * dataset['penalty'].sum(-1) - cw_dict['prize'] * pen.sum(-1)
        return  cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PCTSPDataset(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, cost_weights, edges=None, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size)

        # With beam search we always consider the deterministic case
        state = PCTSPDet.make_state(input, edges, cost_weights, visited_dtype=torch.int64 if compress_mask else torch.uint8)
        return beam_search(state, beam_size, propose_expansions)


class PCTSPDet(PCTSP):
    @staticmethod
    def get_costs(dataset, pi):
        return PCTSP._get_costs(dataset, pi, stochastic=False)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePCTSP.initialize(*args, **kwargs, stochastic=False)


class PCTSPStoch(PCTSP):
    # Stochastic variant of PCTSP, the real (stochastic) prize is only revealed when node is visited
    @staticmethod
    def get_costs(dataset, pi):
        return PCTSP._get_costs(dataset, pi, stochastic=True)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePCTSP.initialize(*args, **kwargs, stochastic=True)


def make_instance(edge_threshold, edge_strategy, args):
    depot, loc, penalty, deterministic_prize, stochastic_prize, *args = args
    ret_dict = {
        'depot': torch.FloatTensor(depot),
        'loc': torch.FloatTensor(loc),
        'penalty': torch.FloatTensor(penalty),
        'deterministic_prize': torch.FloatTensor(deterministic_prize),
        'stochastic_prize': torch.tensor(stochastic_prize)
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


def generate_instance(size, edge_threshold, edge_strategy, graph=None, penalty_factor=3):
    if graph is not None:
        depot, loc = graph
    else:
        depot = torch.rand(2)
        loc = torch.rand(size, 2)

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    penalty_max = MAX_LENGTHS[size] * (penalty_factor) / float(size)
    penalty = torch.rand(size) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = torch.rand(size) * 4 / float(size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = torch.rand(size) * deterministic_prize * 2
    ret_dict = {
        'depot': depot,
        'loc': loc,
        'penalty': penalty,
        'deterministic_prize': deterministic_prize,
        'stochastic_prize': stochastic_prize
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


class PCTSPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, area="riomaior", vertex_strat="mmn", 
                number_edges=0, edge_strat=None, focus_graph=None, focus_size=0, dist_strat=None, waste_type=None, dist_matrix_path=None):
        super(PCTSPDataset, self).__init__()
        assert focus_graph is None or focus_size > 0
        self.data_set = []
        if isinstance(number_edges, str):
            if '.' in number_edges:
                num_edges = float(number_edges)
            else:
                num_edges = int(number_edges)
        else:
            num_edges = number_edges if number_edges is not None else 0

        if focus_graph is not None:
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
            self.dist_matrix = torch.from_numpy(dist_matrix_edges).float() / 100
        else:
            graph = None
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
            self.data = [
                generate_instance(size, num_edges, edge_strat) if focus_size < i
                else generate_instance(size, num_edges, edge_strat, graph)
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