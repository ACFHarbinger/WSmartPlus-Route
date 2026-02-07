"""
BatchBeam class for beam search tracking.
"""

from typing import Any, NamedTuple, Optional

import torch

from .decoding_utils import segment_topk_idx


class BatchBeam(NamedTuple):
    """
    Class that keeps track of a beam for beam search in batch mode.
    """

    score: torch.Tensor  # Current heuristic score of each entry in beam
    state: Any  # To track the state
    parent: Optional[torch.Tensor]
    action: Optional[torch.Tensor]
    batch_size: int  # Can be used for optimizations if batch_size = 1
    device: Any  # Track on which device

    @property
    def ids(self):
        """Returns flattened batch identifiers."""
        return self.state.ids.view(-1)  # Need to flat as state has steps dimension

    def __getitem__(self, key):
        """Allows indexing/slicing of the beam state."""
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                score=self.score[key] if self.score is not None else None,
                state=self.state[key],
                parent=self.parent[key] if self.parent is not None else None,
                action=self.action[key] if self.action is not None else None,
            )
        return tuple.__getitem__(self, key)

    @staticmethod
    def initialize(state):
        """Initializes the beam with the starting state."""
        batch_size = len(state.ids)
        device = state.ids.device
        return BatchBeam(
            score=torch.zeros(batch_size, dtype=torch.float, device=device),
            state=state,
            parent=None,
            action=None,
            batch_size=batch_size,
            device=device,
        )

    def propose_expansions(self):
        """Proposes valid expansions from the current state."""
        mask = self.state.get_mask()
        # Mask always contains a feasible action
        expansions = torch.nonzero(mask[:, 0, :] == 0)
        parent, action = torch.unbind(expansions, -1)
        return parent, action, None

    def expand(self, parent, action, score=None):
        """Expands the beam with chosen actions."""
        return self._replace(
            score=score,  # The score is cleared upon expanding as it is no longer valid, or it must be provided
            state=self.state[parent].update(action),  # Pass ids since we replicated state
            parent=parent,
            action=action,
        )

    def topk(self, k):
        """Selects the top-k beam candidates per batch."""
        idx_topk = segment_topk_idx(self.score, k, self.ids)
        return self[idx_topk]

    def all_finished(self):
        """Checks if all beams have finished decoding."""
        return self.state.all_finished()

    def cpu(self):
        """Moves beam data to CPU."""
        return self.to(torch.device("cpu"))

    def to(self, device):
        """Moves beam data to the specified device."""
        if device == self.device:
            return self
        return self._replace(
            score=self.score.to(device) if self.score is not None else None,
            state=self.state.to(device),
            parent=self.parent.to(device) if self.parent is not None else None,
            action=self.action.to(device) if self.action is not None else None,
        )

    def clear_state(self):
        """Clears the state to save memory."""
        return self._replace(state=None)

    def size(self):
        """Returns the current beam size (number of active paths)."""
        if self.state is None:
            return 0
        return self.state.ids.size(0)
