
import os
import torch
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from logic.src.utils.data_utils import load_focus_coords, generate_waste_prize
from logic.src.utils.graph_utils import get_edge_idx_dist, get_adj_knn, adj_to_idx
from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.network import compute_distance_matrix, apply_edges
from logic.src.utils.definitions import MAX_WASTE, VEHICLE_CAPACITY
from .state_scwcvrp import StateSCWCVRP
from logic.src.utils.beam_search import beam_search


class SCWCVRP(object):
    NAME = 'scwvrp'  # Stochastic Capacitated Waste Collection Vehicle Routing Problem
    VEHICLE_CAPACITY = 1.0

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        batch_size, graph_size = dataset['real_waste'].size()

        if pi.size(-1) == 1:
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            overflows = torch.sum(dataset['real_waste'] >= dataset['max_waste'][:, None], dim=-1)
            cost = overflows if cw_dict is None \
            else cw_dict['overflows'] * overflows
            c_dict = {'overflows': overflows, 'length': torch.zeros_like(cost), 'waste': torch.zeros_like(cost), 'total': cost}
            return cost, c_dict, None

        # Helper method to calculate costs using Real Waste
        # 1. Reset capacity at depot.
        # 2. Check capacity constraints with Real Waste.
        
        # Prepare Waste tensor with depot (-Capacity)
        wastes = torch.cat(
            (
                torch.full_like(dataset['real_waste'][:, :1], -VEHICLE_CAPACITY),
                dataset['real_waste']
            ),
            1
        )
        
        # In SCWCVRP, if we visit a node and its real waste exceeds remaining capacity, 
        # we still collect what we can? Or is it strict capacity?
        # CWCVRP implementation assumes rigid capacity constraint satisfaction in generating valid tours.
        # But here, since we don't know real waste, we might violate it.
        # Standard VRP with stochastic demands usually adds a distinct penalty for route failure (recourse).
        # However, following the CWCVRP implementation style: we just calculate used capacity.
        # But wait, CWCVRP `get_costs` asserts `used_cap <= VEHICLE_CAPACITY`.
        # If our policy generated a route that violates REAL capacity (which it couldn't see), 
        # asserting here would fail evaluation.
        # Instead, we should probably allow violation but maybe penalize? 
        # OR: The standard "Split Delivery" style (SDWCVRP) logic where "d = min(waste, capacity - used)".
        # Let's adopt the SDWCVRP logic: we collect up to capacity.
        
        rng = torch.arange(batch_size, out=wastes.data.new().long())
        used_cap = torch.zeros_like(dataset['real_waste'][:, 0])
        
        # Mutable copy of wastes to track remaining waste at nodes
        current_node_wastes = wastes.clone()

        for a in pi.transpose(0, 1):
            # Calculate how much we CAN collect
            # waste at node a vs remaining capacity
            # Note: depot (0) has -Capacity, so min will be -Capacity (reset mechanism)
            
            # If a is depot, desired_collection is -1 (full reset). 
            # If a is node, desired is waste.
            
            val_at_node = current_node_wastes[rng, a]
            
            # Remaining capacity
            remaining_cap = VEHICLE_CAPACITY - used_cap
            
            # Amount to load:
            # If val_at_node < 0 (Depot): we "load" val_at_node (which reduces used_cap)
            # If val_at_node > 0: we load min(val, remaining)
            
            # Logic from SDWCVRP:
            # d = torch.min(wastes[rng, a], VEHICLE_CAPACITY - used_cap)
            # wastes[rng, a] -= d
            # used_cap += d
            # used_cap[a == 0] = 0
            
            # Note: SDWCVRP implementation uses logic that handles depot reset explicitly via used_cap[a==0]=0
            
            d = torch.min(val_at_node, remaining_cap)
            # But if a==0, val_at_node is -1. remaining_cap is >= 0. min is -1.
            # So current_node_wastes[depot] -= (-1) -> adds 1? No.
            # SDWCVRP seems to rely on depot having large negative value?
            
            # Let's stick closer to CWCVRP but robust to overflow.
            # CWCVRP:
            # d = waste_with_depot.gather(1, pi) -> gather visits
            # used_cap += d
            # assert <= capacity
            
            # Here:
            d = current_node_wastes[rng, a]
            
            # If d > remaining_cap, we have a failure/overflow of truck.
            # In stochastic VRP, usually you return to depot and come back (recourse).
            # Simplified: we just cap the collection at remaining capacity?
            # Or we let it exceed and count is as "Recourse Cost"?
            # Given the lack of specific Recourse logic in the prompt, strict capacity might be assumed 
            # OR we assume "collect as much as possible".
            # "Only after going to the location does it know the true waste values".
            
            # I will assume "Collect as much as possible, leave the rest".
            
            actual_collected = d.clone()
            # Mask for non-depot
            is_node = (a != 0)
            
            # Check capacity violation
            violation_mask = is_node & (d > remaining_cap)
            
            # Clip actual collected to remaining capacity
            actual_collected[violation_mask] = remaining_cap[violation_mask]
            
            # Update remaining waste at node
            current_node_wastes[rng, a] -= actual_collected
            
            # Update used capacity
            used_cap += actual_collected
            
            # Reset at depot
            used_cap[~is_node] = 0
            
            # Note: what if we return to the node? SCWCVRP (like PCTSP) assumes duplicates masked out usually?
            # StatePCTSP: "Make sure each node visited once at most". 
            
        # Calculate Overflows (Uncollected Waste > Max)
        # Note: current_node_wastes at index 0 is garbage, ignore.
        
        overflows = torch.sum(current_node_wastes[:, 1:] >= dataset['max_waste'][:, None], dim=-1)
        
        # Length Calculation
        if dist_matrix is not None:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst = torch.max(dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device), dim=1).indices
            length = dist_matrix[0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0] + dists.sum(dim=1) + dist_matrix[0, 0, src_vertices[:, 0]]
        else:
            loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            d_coord = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
            length = (
                (d_coord[:, 1:] - d_coord[:, :-1]).norm(p=2, dim=-1).sum(1)
                + (d_coord[:, 0] - dataset['depot']).norm(p=2, dim=-1)
                + (d_coord[:, -1] - dataset['depot']).norm(p=2, dim=-1)
            )

        # Waste collected = Total Initial Real Waste - Remaining Real Waste
        waste = dataset['real_waste'].sum(dim=-1) - current_node_wastes[:, 1:].sum(dim=-1)
        
        cost = overflows + length - waste if cw_dict is None \
        else cw_dict['overflows'] * overflows + cw_dict['length'] * length - cw_dict['waste'] * waste
        c_dict = {'overflows': overflows, 'length': length, 'waste': waste, 'total': cost}
        return cost, c_dict, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return SWCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSCWCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, cost_weights, edges=None, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        fixed = model.precompute_fixed(input)
        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size)
        state = SCWCVRP.make_state(input, edges, cost_weights, visited_dtype=torch.int64 if compress_mask else torch.uint8)
        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, edge_threshold, edge_strategy, distribution, bins, *args, graph=None, noise_mean=0.0, noise_variance=0.0):
    if graph is not None:
        depot, loc = graph
    else:
        loc = torch.FloatTensor(size, 2).uniform_(0, 1)
        depot = torch.FloatTensor(2).uniform_(0, 1)
        
    # Generate Real Waste    
    real_waste = torch.from_numpy(generate_waste_prize(size, distribution, (depot, loc), 1, bins)).float()
    
    # Generate Noisy Waste
    if noise_variance > 0:
        noise = torch.normal(mean=noise_mean, std=noise_variance**0.5, size=real_waste.size())
        noisy_waste = (real_waste + noise).clamp(min=0.0, max=MAX_WASTE)
    else:
        noisy_waste = real_waste
    
    ret_dict = {
        'loc': loc,
        'depot': depot,
        'real_waste': real_waste,
        'noisy_waste': noisy_waste,
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


class SWCVRPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution='unif', area="riomaior", vertex_strat="mmn", 
                number_edges=0, edge_strat=None, focus_graph=None, focus_size=0, dist_strat=None, waste_type=None, dist_matrix_path=None,
                noise_mean=0.0, noise_variance=0.0):
        super(SWCVRPDataset, self).__init__()

        self.noise_mean = noise_mean
        self.noise_variance = noise_variance

        if isinstance(number_edges, str):
            if '.' in number_edges:
                num_edges = float(number_edges)
            else:
                num_edges = int(number_edges)
        else:
            num_edges = number_edges if number_edges is not None else 0

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
            if distribution in ['gamma', 'emp']:
                bins = Bins(size, os.path.join(os.getcwd(), "data", "wsr_simulator"), distribution, area=area, indices=idx[0], waste_type=waste_type,
                           noise_mean=self.noise_mean, noise_variance=self.noise_variance)
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
                # Data is expected to be list of tuples/dicts
                self.data = []
                for item in tqdm(data[offset:offset+num_samples]):
                    depot, loc, real_waste, noisy_waste, max_waste = item
                    
                    real_waste_t = torch.FloatTensor(real_waste)
                    noisy_waste_t = torch.FloatTensor(noisy_waste)
                    
                    if real_waste_t.dim() > 1:
                        real_waste_t = real_waste_t[0]
                    if noisy_waste_t.dim() > 1:
                        noisy_waste_t = noisy_waste_t[0]
                        
                    instance = {
                        'depot': torch.FloatTensor(depot),
                        'loc': torch.FloatTensor(loc),
                        'real_waste': real_waste_t,
                        'noisy_waste': noisy_waste_t,
                        'max_waste': torch.FloatTensor(max_waste),
                    }
                    if num_edges > 0:
                         distance_matrix = squareform(pdist(instance['loc'], metric='euclidean'))
                         if edge_strat == 'dist':
                            instance['edges'] = torch.tensor(get_edge_idx_dist(distance_matrix, num_edges)).to(dtype=torch.long)
                         elif edge_strat == 'knn':
                            neg_adj_matrix = get_adj_knn(distance_matrix, num_edges)
                            instance['edges'] = torch.tensor(adj_to_idx(neg_adj_matrix)).to(dtype=torch.long)
                    self.data.append(instance)
                    
        else:
            print("Generating data...")
            # Generation loop
            self.data = []
            for i in tqdm(range(num_samples)):
                current_graph = None
                if graph is not None and i < focus_size:
                    current_graph = (graph[0][i, :], graph[1][i, :, :])

                ret_dict = generate_instance(size, num_edges, edge_strat, distribution, bins, graph=current_graph,
                                             noise_mean=self.noise_mean, noise_variance=self.noise_variance)
                self.data.append({
                    'depot': ret_dict['depot'],
                    'loc': ret_dict['loc'],
                    'real_waste': ret_dict['real_waste'],
                    'noisy_waste': ret_dict['noisy_waste'],
                    'max_waste': ret_dict['max_waste'],
                    'edges': ret_dict.get('edges') 
                })
             
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

