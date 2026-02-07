from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .base import DecodingStrategy
from .batch_beam import BatchBeam
from .decoding_utils import backtrack


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
