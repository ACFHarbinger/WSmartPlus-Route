"""
Base classes and utilities for routing problems, datasets, and states.
"""
import torch
from logic.src.utils.beam_search import beam_search as beam_search_func
from logic.src.utils.boolmask import mask_long2bool



class BaseProblem(object):
    """
    Base class for routing problems (WCVRP, VRPP, SCWCVRP).
    Consolidates shared logic for cost calculation, tour validation, and beam search.
    """

    @staticmethod
    def validate_tours(pi):
        """
        Validates that tours are valid (contain 0 to n-1, no duplicates except depot).
        
        Args:
            pi (Tensor): The sequence of visited nodes (batch_size, tour_length).
        """
        if pi.size(-1) == 1:
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            return True

        sorted_pi = pi.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])
        ).all(), "Duplicate nodes found in tour"
        return True

    @staticmethod
    def get_tour_length(dataset, pi, dist_matrix=None):
        """
        Calculates the tour length (L2 norm or from distance matrix).
        
        Args:
            dataset (dict): Dataset containing 'depot', 'loc', etc.
            pi (Tensor): tour sequence.
            dist_matrix (Tensor, optional): Precomputed distance matrix.
            
        Returns:
            Tensor: length of each tour in the batch.
        """
        if pi.size(-1) == 1:
            return torch.zeros(pi.size(0), device=pi.device)

        if dist_matrix is not None:
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask = dst_vertices != 0
            pair_mask = (src_vertices != 0) & (dst_mask)
            dists = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            
            last_dst = torch.max(
                dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
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
                + dist_matrix[0, 0, pi[:, 0]]
            )
        else:
            loc_with_depot = torch.cat((dataset["depot"][:, None, :], dataset["loc"]), 1)
            d = loc_with_depot.gather(
                1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1))
            )
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)
                + (d[:, 0] - dataset["depot"]).norm(p=2, dim=-1)
                + (d[:, -1] - dataset["depot"]).norm(p=2, dim=-1)
            )
        return length

    @classmethod
    def beam_search(
        cls,
        input,
        beam_size,
        cost_weights,
        edges=None,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
        **kwargs
    ):
        """
        Standardized beam search implementation.
        """
        assert model is not None, "Provide model"
        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            """
            Proposes next nodes for the beam based on model scores.
            """
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = cls.make_state(
            input,
            edges,
            cost_weights,
            visited_dtype=torch.int64 if compress_mask else torch.uint8,
            **kwargs
        )
        return beam_search_func(state, beam_size, propose_expansions)


class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for routing datasets.
    """
    def __init__(self):
        """Initializes the dataset."""
        super(BaseDataset, self).__init__()
        self.data = []
        self.size = 0

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Returns the data sample at the given index."""
        return self.data[idx]

    def __setitem__(self, key, values):
        """Updates a specific field across all samples in the dataset."""
        def __update_item(inst, k, v):
            """Internal helper to update a dictionary item."""
            inst[k] = v
            return inst
        self.data = [__update_item(x, key, val) for x, val in zip(self.data, values)]


class BaseState(object):
    """
    Collection of common state functionality.
    To be injected into NamedTuple classes.
    """

    @property
    def visited(self):
        """Returns a boolean mask of visited nodes."""
        if not hasattr(self, "visited_"):
            return None
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self):
        """Calculates the Euclidean distance matrix between all coordinates."""
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(
            p=2, dim=-1
        )

    def all_finished(self):
        """Checks if all instances in the batch have returned to the depot."""
        return self.i.item() > 0 and (self.prev_a == 0).all()

    def get_current_node(self):
        """Returns the current node index."""
        return self.prev_a

    def get_edges_mask(self):
        """Returns a mask based on graph edges for the current node."""
        batch_size, n_coords, _ = self.coords.size()
        if self.i.item() == 0:
            return torch.zeros(
                batch_size, 1, n_coords, dtype=torch.uint8, device=self.coords.device
            )
        else:
            return self.edges.gather(
                1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_coords)
            )

    def get_edges(self):
        """Returns the graph edge indices."""
        return self.edges

    def construct_solutions(self, actions):
        """Returns the sequences of actions as solutions."""
        return actions

    @staticmethod
    def initialize_common(input, visited_dtype=torch.uint8):
        """
        Computes common initialization fields.
        """
        depot = input["depot"]
        loc = input["loc"]
        batch_size, n_loc, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), -2)
        
        ids = torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None]
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        
        if visited_dtype == torch.uint8:
            visited_ = torch.zeros(batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device)
        else:
            visited_ = torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)
            
        lengths = torch.zeros(batch_size, 1, device=loc.device)
        cur_coord = depot[:, None, :]
        i = torch.zeros(1, dtype=torch.int64, device=loc.device)
        
        return {
            "coords": coords,
            "ids": ids,
            "prev_a": prev_a,
            "visited_": visited_,
            "lengths": lengths,
            "cur_coord": cur_coord,
            "i": i,
            "batch_size": batch_size,
            "n_loc": n_loc
        }


def refactor_state(cls):
    """
    Decorator to inject BaseState methods into a NamedTuple class.
    """
    for name, attr in BaseState.__dict__.items():
        if not name.startswith("__") and name != "initialize_common":
            setattr(cls, name, attr)
    return cls

