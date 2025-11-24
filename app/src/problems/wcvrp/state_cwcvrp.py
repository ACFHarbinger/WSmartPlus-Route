import torch

from typing import NamedTuple
from app.src.utils.boolmask import mask_long2bool, mask_long_scatter


class StateCWCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    waste: torch.Tensor

    # Cost function weights
    w_waste: float
    w_length: float
    w_overflows: float

    # Maximum amount of waste before bin is considered to be overflowing
    max_waste: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and waste tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_overflows: torch.Tensor
    cur_total_waste: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    edges: torch.Tensor
    dist_matrix: torch.Tensor

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                cur_overflows=self.cur_overflows[key],
                cur_total_waste=self.cur_total_waste[key]
            )
        return self[key]

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)
    @staticmethod
    def initialize(input, edges, cost_weights=None, dist_matrix=None, visited_dtype=torch.uint8):
        depot = input['depot']
        loc = input['loc']
        waste = input['waste']
        max_waste = input['max_waste']
        batch_size, n_loc, _ = loc.size()
        return StateCWCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            waste=waste,
            max_waste=max_waste[:, None],
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=waste.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            cur_overflows=torch.sum((input['waste'] > max_waste[:, None]), dim=-1),
            cur_total_waste=torch.zeros(batch_size, 1, device=loc.device),
            #cur_waste_lost=torch.sum((waste - max_waste[:, None]).clamp(min=0), dim=-1),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            w_waste=1 if cost_weights is None else cost_weights['waste'],
            w_overflows=1 if cost_weights is None else cost_weights['overflows'],
            w_length=1 if cost_weights is None else cost_weights['length'],
            edges=edges,
            dist_matrix=dist_matrix
        )

    def get_final_cost(self):
        assert self.all_finished()
        length_cost = self.w_length * self.lengths + self.w_length * (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)
        return self.w_overflows * self.cur_overflows + length_cost + self.w_waste * self.cur_total_waste

    def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.waste.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.waste[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)].clamp(max=self.max_waste[self.ids, 0])
        cur_total_waste = self.cur_total_waste + selected_demand

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()
        
        # Update the number of overflows
        cur_overflows = self.cur_overflows - torch.sum((self.waste[self.ids, selected] >= self.max_waste), dim=-1)

        # Update the amount of waste overflowing from the bins
        #cur_waste_lost = self.cur_waste_lost - torch.sum((self.waste[self.ids, selected] - self.max_waste).clamp(min=0), dim=-1)

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, cur_overflows=cur_overflows, cur_total_waste=cur_total_waste, i=self.i+1
        )

    def all_finished(self):
        # We dont need to visit all bins to end state, just arrive at the depot
        return self.i.item() >= self.waste.size(-1) and (self.prev_a == 0).all() # self.visited.all() 

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.waste.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.waste[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def get_edges_mask(self):
        batch_size, n_coords, _ = self.coords.size()
        if self.i.item() == 0:
            return torch.zeros(batch_size, 1, n_coords, dtype=torch.uint8, device=self.coords.device)
        else:
            return self.edges.gather(1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_coords))
    
    def get_edges(self):
        return self.edges
    
    def construct_solutions(self, actions):
        return actions