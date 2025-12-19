import torch
import torch.nn.functional as F

from typing import NamedTuple
from logic.src.utils.boolmask import mask_long2bool, mask_long_scatter


class StateWCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    waste: torch.Tensor

    # Cost function weights
    #w_lost: float
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
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_total_waste: torch.Tensor
    cur_overflows: torch.Tensor
    #cur_waste_lost: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    edges: torch.Tensor
    dist_matrix: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                cur_total_waste=self.cur_total_waste[key],
                cur_overflows=self.cur_overflows[key],
                #cur_waste_lost=self.cur_waste_lost[key]
            )
        return self[key]

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)
    @staticmethod
    def initialize(input, edges, cost_weights=None, dist_matrix=None, visited_dtype=torch.uint8, hrl_mask=None):
        depot = input['depot']
        loc = input['loc']
        waste = input['waste']
        max_waste = input['max_waste']

        batch_size, n_loc, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), -2)

        # Initialize visited mask
        if visited_dtype == torch.uint8:
            visited_ = torch.zeros(
                batch_size, 1, n_loc + 1,
                dtype=torch.uint8, device=loc.device
            )
            if hrl_mask is not None:          
                # hrl_mask is (Batch, N).
                # visited_ is (Batch, 1, N+1). Index 0 is Depot.
                # Mark HRL masked nodes as visited (1)
                if hrl_mask.dim() == 2:
                    if hrl_mask.size(1) != n_loc:
                        print(f"CRITICAL ERROR: HRL Mask width {hrl_mask.size(1)} != n_loc {n_loc}")
                    visited_[:, 0, 1:] = hrl_mask.to(torch.uint8)
                else:
                    # If mask has extra dims
                    visited_[:, 0, 1:] = hrl_mask.squeeze().to(torch.uint8)
        else:
            # Compressed mask logic (int64 bitmask)
            visited_ = torch.zeros(batch_size, 1, (n_loc + 1 + 63) // 64, dtype=torch.int64, device=loc.device)
            # NOTE: Compressed mask HRL not implemented yet, assumes uint8 default
            if hrl_mask is not None:
                print("Warning: HRL Mask with compressed visited mask not implemented")

        return StateWCVRP(
            coords=coords,
            waste=F.pad(waste, (1, 0), mode='constant', value=0),  # add 0 for depot
            max_waste=max_waste[:, None],
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=visited_,
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            cur_total_waste=torch.zeros(batch_size, 1, device=loc.device),
            cur_overflows=torch.sum((input['waste'] >= max_waste[:, None]), dim=-1),
            #cur_waste_lost=torch.sum((waste - max_waste[:, None]).clamp(min=0), dim=-1),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            w_waste=1 if cost_weights is None else cost_weights['waste'],
            w_overflows=1 if cost_weights is None else cost_weights['overflows'],
            w_length=1 if cost_weights is None else cost_weights['length'],
            #w_lost=1 if cost_weights is None else cost_weights['lost'],
            edges=edges,
            dist_matrix=dist_matrix
        )

    def get_final_cost(self):

        assert self.all_finished()

        # The cost is the negative of the collected waste since we want to maximize collected waste
        return self.w_overflows * self.cur_overflows + self.w_length * self.lengths - self.w_waste * self.cur_total_waste #+ self.w_lost * self.cur_waste_lost

    def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        #lengths = self.lengths + self.dist_matrix[0, self.prev_a, selected]

        # Add the collected waste
        cur_total_waste = self.cur_total_waste + self.waste[self.ids, selected].clamp(max=self.max_waste[self.ids, 0])

        # Update the number of overflows
        cur_overflows = self.cur_overflows - torch.sum((self.waste[self.ids, selected] >= self.max_waste[self.ids, 0]), dim=-1)

        # Update the amount of waste overflowing from the bins
        #cur_waste_lost = self.cur_waste_lost - torch.sum((self.waste[self.ids, selected] - self.max_waste).clamp(min=0), dim=-1)

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a, visited_=visited_, lengths=lengths, cur_coord=cur_coord,
            cur_total_waste=cur_total_waste, cur_overflows=cur_overflows, i=self.i+1
        )

    def all_finished(self):
        # All must be returned to depot (and at least 1 step since at start also prev_a == 0)
        # This is more efficient than checking the mask
        return self.i.item() > 0 and (self.prev_a == 0).all()
        # return self.visited[:, :, 0].all()  # If we have visited the depot we're done

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a
    
    def get_remaining_overflows(self):
        return self.cur_overflows.unsqueeze(-1)
    
    def get_current_efficiency(self):
        efficiency = self.cur_total_waste / self.lengths
        return torch.nan_to_num(efficiency, nan=0.0)

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # If the depot has already been visited then we cannot visit anymore
        visited_ = self.visited > 0
        mask = visited_ | visited_[:, :, 0:1]
        # Depot can always be visited
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        mask[:, :, 0] = 0
        return mask
    
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