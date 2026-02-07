from typing import Any, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .base import DecodingStrategy
from .decoding_utils import backtrack, segment_topk_idx


class BeamSearch(DecodingStrategy):
    """
    Beam search decoding: maintain top-k partial solutions.

    Note: Beam search requires special handling in the policy forward pass.
    This implementation provides the step function for scoring candidates.
    """

    def __init__(self, beam_width: int = 5, **kwargs):
        """
        Initialize BeamSearch decoding.

        Args:
            beam_width: Number of beams to maintain.
            **kwargs: Passed to super class.
        """
        super().__init__(**kwargs)
        self.beam_width = beam_width

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k actions for beam search.

        Returns:
            Tuple of (top_k_actions, top_k_log_probs, entropy) each with shape [batch, beam_width]
        """
        logits = self._process_logits(logits, mask)
        log_probs = F.log_softmax(logits, dim=-1)

        # Get top-k
        top_log_probs, top_actions = torch.topk(log_probs, self.beam_width, dim=-1)

        # Placeholder entropy for beam search (complex to define per step)
        entropy = torch.zeros_like(top_log_probs)

        return top_actions, top_log_probs, entropy


def beam_search(*args, **kwargs):
    """
    Orchestrates beam search decoding on a batch of states.

    Args:
        *args: positional args passed to _beam_search
        **kwargs: keyword args passed to _beam_search

    Returns:
        tuple: (score, solutions, cost, ids, batch_size)
    """
    beams, final_state = _beam_search(*args, **kwargs)
    return get_beam_search_results(beams, final_state)


def get_beam_search_results(beams, final_state):
    """
    Reconstructs solutions from the final beam state.

    Args:
        beams (list): List of BatchBeam objects from each step.
        final_state (State): The final state after decoding.

    Returns:
        tuple: (score, solutions, cost, ids, batch_size)
    """
    beam = beams[-1]  # Final beam
    if final_state is None:
        return None, None, None, None, beam.batch_size

    # First state has no actions/parents and should be omitted when backtracking
    actions = [beam.action for beam in beams[1:]]
    parents = [beam.parent for beam in beams[1:]]

    solutions = final_state.construct_solutions(backtrack(parents, actions))
    return (
        beam.score,
        solutions,
        final_state.get_final_cost()[:, 0],
        final_state.ids.view(-1),
        beam.batch_size,
    )


def _beam_search(state, beam_size, propose_expansions=None, keep_states=False):
    """
    Internal beam search execution.

    Args:
        state (State): Initial state.
        beam_size (int): Width of the beam.
        propose_expansions (callable, optional): Custom expansion logic.
        keep_states (bool, optional): Whether to store all intermediate states.

    Returns:
        tuple: (beams, final_state)
    """
    beam = BatchBeam.initialize(state)

    # Initial state
    beams = [beam if keep_states else beam.clear_state()]

    # Perform decoding steps
    while not beam.all_finished():
        # Use the model to propose and score expansions
        parent, action, score = beam.propose_expansions() if propose_expansions is None else propose_expansions(beam)
        if parent is None:
            return beams, None

        # Expand and update the state according to the selected actions
        beam = beam.expand(parent, action, score=score)

        # Get topk
        beam = beam.topk(beam_size)

        # Collect output of step
        beams.append(beam if keep_states else beam.clear_state())

    # Return the final state separately since beams may not keep state
    return beams, beam.state


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
