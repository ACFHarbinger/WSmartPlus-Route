import os
import json
import torch
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .state_pdp import StatePDP
from logic.src.utils.beam_search import beam_search
from logic.src.utils.data_utils import load_focus_coords
from scipy.spatial.distance import pdist, squareform
from logic.src.utils.graph_utils import get_edge_idx_dist, get_adj_knn, adj_to_idx
from logic.src.pipeline.simulator.network import compute_distance_matrix, apply_edges


class PDP(object):
    NAME = 'pdp'  # Capacitated Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):  # pi:[batch_size, graph_size]
        assert (pi[:, 0]==0).all(), "not starting at depot"
        assert (torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) == pi.data.sort(1)[0]).all(), "not visiting all nodes"

        visited_time = torch.argsort(pi, 1)  # pickup的index < 对应的delivery的index
        assert (visited_time[:, 1:pi.size(1) // 2 + 1] < visited_time[:, pi.size(1) // 2 + 1:]).all(), "deliverying without pick-up"
        
        if dist_matrix is not None:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst = torch.max(dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device), dim=1).indices
            length = dist_matrix[0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0] + dists.sum(dim=1) + dist_matrix[0, 0, src_vertices[:, 0]]
        else:
            # dataset['depot']: [batch_size, 2], dataset['loc']: [batch_size, graph_size, 2]
            dataset = torch.cat([dataset['depot'].reshape(-1, 1, 2), dataset['loc']], dim = 1)  # [batch, graph_size+1, 2]
            d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))  # [batch, graph_size+1, 2]
            # d[:, :-1] do not include -1
            length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
        cost = length if cw_dict is None else length * cw_dict['length']
        return cost, None


    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDP.initialize(*args, **kwargs)
        
    @staticmethod
    def beam_search(input, beam_size, cost_weights, edges=None, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)
        state = PDP.make_state(input, edges, cost_weights, visited_dtype=torch.int64 if compress_mask else torch.uint8)
        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size)

        return beam_search(state, beam_size, propose_expansions)

def make_instance(edge_threshold, edge_strategy, args):
    depot, loc, *args = args
    # grid_size = 1
    # if len(args) > 0:
    #     depot_types, customer_types, grid_size = args
    ret_dict = {
        'loc': torch.tensor(loc, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float)
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


def generate_instance(size, edge_threshold, edge_strategy, graph):
    if graph is not None:
        ret_dict = {
            'loc': graph[1],
            'depot': graph[0]
        }
    else:
        ret_dict = {
            'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
            'depot': torch.FloatTensor(2).uniform_(0, 1)
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


class PDPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, area="riomaior", vertex_strat="mmn", 
                number_edges=0, edge_strat=None, focus_graph=None, focus_size=0, dist_strat=None, waste_type=None, dist_matrix_path=None):
        super(PDPDataset, self).__init__()
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