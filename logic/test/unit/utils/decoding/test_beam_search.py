"""Unit tests for beam_search.py."""

from typing import Any, cast
from unittest.mock import MagicMock

import torch
from logic.src.utils.decoding import (
    BatchBeam,
    CachedLookup,
    _beam_search,
    backtrack,
    beam_search,
    get_beam_search_results,
    segment_topk_idx,
)


def test_segment_topk_idx_single_group():
    """Test topk when there is only one group (batch_size=1)."""
    x = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    ids = torch.zeros(5, dtype=torch.long)
    k = 3

    idx = segment_topk_idx(x, k, ids)
    # Top 3 are 5.0, 4.0, 3.0 -> indices 1, 3, 2
    expected = torch.tensor([1, 3, 2])
    assert torch.equal(idx, expected)

def test_segment_topk_idx_multi_group():
    """Test topk with multiple segments of varying sizes."""
    # Batch 0: [10, 20, 30] -> top 2: [30, 20] -> idx [2, 1]
    # Batch 1: [5, 2]       -> top 2: [5, 2]   -> idx [3, 4]
    # Batch 2: [100, 50, 75] -> top 2: [100, 75] -> idx [5, 7]
    x = torch.tensor([10.0, 20.0, 30.0, 5.0, 2.0, 100.0, 50.0, 75.0])
    ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
    k = 2

    idx = segment_topk_idx(x, k, ids)

    # Sort order per group:
    # G0 sorted: idx 2, 1, 0
    # G1 sorted: idx 3, 4
    # G2 sorted: idx 5, 7, 6

    # top 2 per group: 2, 1, 3, 4, 5, 7
    expected = torch.tensor([2, 1, 3, 4, 5, 7])
    assert torch.equal(idx, expected)

def test_segment_topk_idx_k_larger_than_segment():
    """Test topk when k is larger than the segment size."""
    x = torch.tensor([1.0, 2.0])
    ids = torch.zeros(2, dtype=torch.long)
    k = 5

    idx = segment_topk_idx(x, k, ids)
    assert torch.equal(idx, torch.tensor([1, 0]))

def test_backtrack():
    """Test path reconstruction through parent pointers."""
    # Step 1:
    parents1 = torch.tensor([0, 1]) # Unused for first backtrack step usually
    actions1 = torch.tensor([1, 2]) # Batch 0 took 1, Batch 1 took 2

    # Step 2:
    # Batch 0 Beams: B0_0 (child of A1_0), B0_1 (child of A1_0)
    # Batch 1 Beams: B1_0 (child of A1_1), B1_1 (child of A1_1)
    parents2 = torch.tensor([0, 0, 1, 1]) # Indices into previous beam
    actions2 = torch.tensor([10, 11, 20, 21])

    res = backtrack([parents1, parents2], [actions1, actions2])

    # Expected result: [TotalBeams, SeqLen]
    # Beam 0: parent 0 of step 1 was 1. Seq: [1, 10]
    # Beam 1: parent 0 of step 1 was 1. Seq: [1, 11]
    # Beam 2: parent 1 of step 1 was 2. Seq: [2, 20]
    # Beam 3: parent 1 of step 1 was 2. Seq: [2, 21]

    expected = torch.tensor([
        [1, 10],
        [1, 11],
        [2, 20],
        [2, 21]
    ])
    assert torch.equal(res, expected)

def test_cached_lookup():
    """Test CachedLookup utility."""
    data = torch.tensor([10, 20, 30, 40])
    lookup = CachedLookup(data)

    # First access
    idx1 = torch.tensor([0, 2])
    val1 = cast(Any, lookup[idx1])
    assert torch.equal(val1, torch.tensor([10, 30]))
    assert lookup.key is idx1

    # Access with same key (should return cached)
    val2 = lookup[idx1]
    assert val2 is val1

    # Access with different key
    idx2 = torch.tensor([1, 3])
    val3 = cast(Any, lookup[idx2])
    assert torch.equal(val3, torch.tensor([20, 40]))
    assert lookup.key is idx2

def test_batch_beam_lifecycle():
    """Test BatchBeam initialization and basic properties."""
    state = MagicMock()
    state.ids = torch.tensor([0, 1, 2])
    state.get_mask = MagicMock(return_value=torch.zeros((3, 1, 5))) # 3 beams, 1 step, 5 nodes
    state.all_finished = MagicMock(return_value=False)

    # Initialize
    beam = BatchBeam.initialize(state)
    assert beam.batch_size == 3
    assert cast(torch.Tensor, beam.score).shape == (3,)
    assert torch.all(cast(torch.Tensor, beam.score) == 0)
    assert beam.parent is None

    # Propose expansions
    # state.get_mask() returns [Beams, Steps, Nodes]
    # mask = 0 means valid.
    # nonzero on mask[:, 0, :] == 0
    # If all 0, and 3 beams, 5 nodes -> 15 expansions
    parent, action, score = beam.propose_expansions()
    assert len(parent) == 15
    assert len(action) == 15
    assert score is None

    # Expand
    # mock update to return a new state
    new_state = MagicMock()
    state.__getitem__.return_value = state # Simplification for mock
    state.update.return_value = new_state

    next_beam = beam.expand(parent, action, score=torch.linspace(0, 1, 15))
    assert next_beam.state == new_state
    assert torch.equal(cast(torch.Tensor, next_beam.parent), parent)
    assert torch.equal(cast(torch.Tensor, next_beam.action), action)
    assert cast(torch.Tensor, next_beam.score).shape == (15,)

def test_get_beam_search_results_empty():
    """Test get_beam_search_results with None state."""
    beam = MagicMock(batch_size=5)
    res = get_beam_search_results([beam], None)
    assert res == (None, None, None, None, 5)


class MockState:
    """Minimal mock state for beam search integration testing."""
    def __init__(self, ids, step=0, total_steps=2, parent_idx=None):
        self.ids = ids
        self.step = step
        self.total_steps = total_steps
        self.parent_idx = parent_idx # Index in parent beam

    def get_mask(self):
        # [Beams, 1, Nodes]
        # 2 nodes, node 0 always valid until total_steps
        mask = torch.zeros((len(self.ids), 1, 2))
        if self.step >= self.total_steps:
            mask[:] = 1 # All finished
        return mask

    def update(self, action):
        return MockState(
            self.ids,
            step=self.step + 1,
            total_steps=self.total_steps,
            parent_idx=torch.arange(len(self.ids))
        )

    def all_finished(self):
        return self.step >= self.total_steps

    def construct_solutions(self, actions):
        return actions # Just return actions as solutions

    def get_final_cost(self):
        # [Beams, 1]
        return torch.ones((len(self.ids), 1)) * 10.0

    def __getitem__(self, key):
        if isinstance(key, torch.Tensor):
            return MockState(self.ids[key], self.step, self.total_steps, self.parent_idx)
        return self


def test_beam_search_integration():
    """Test full _beam_search execution on a toy problem."""
    # Batch size 2, 2 steps
    ids = torch.tensor([0, 1])
    initial_state = MockState(ids, total_steps=2)

    # beam_size = 2
    # Provide expansion scores to avoid None score issues in topk
    def mock_propose(beam):
        parent, action, _ = beam.propose_expansions()
        score = torch.linspace(0, 1, len(parent))
        return parent, action, score

    beams, final_state = _beam_search(initial_state, beam_size=2, propose_expansions=mock_propose)

    assert len(beams) == 3 # Initial + 2 steps
    assert final_state is not None
    assert final_state.step == 2
    assert final_state.all_finished()

    # Test top-level beam_search orchestrator
    score, solutions, cost, res_ids, batch_size = beam_search(
        initial_state, beam_size=2, propose_expansions=mock_propose
    )

    assert batch_size == 2
    assert cost is not None
    assert cast(torch.Tensor, cost).shape == (4,)
    assert solutions is not None
    assert solutions.shape[0] == 4
    assert solutions.shape[1] == 2
    assert res_ids is not None
    assert res_ids.shape == (4,)


def test_batch_beam_device_movement():
    """Test to() and cpu() methods of BatchBeam."""
    state = MagicMock()
    state.ids = torch.tensor([0, 1])
    state.to.return_value = state

    beam = BatchBeam(
        score=torch.tensor([1.0, 2.0]),
        state=state,
        parent=torch.tensor([0, 0]),
        action=torch.tensor([1, 2]),
        batch_size=2,
        device=torch.device("cpu")
    )

    # Move to same device
    assert beam.to(torch.device("cpu")) is beam

    # Mock behavior for "cuda" if available or just another device
    new_device = torch.device("cpu") # simulating move
    moved = beam.to(new_device)
    assert moved.device == new_device

    cpu_beam = moved.cpu()
    assert cpu_beam.device == torch.device("cpu")


def test_batch_beam_slicing():
    """Test slicing/indexing of BatchBeam."""
    state = MagicMock()
    state.__getitem__.return_value = "sliced_state"

    beam = BatchBeam(
        score=torch.tensor([1.0, 2.0, 3.0]),
        state=state,
        parent=torch.tensor([0, 0, 1]),
        action=torch.tensor([1, 2, 3]),
        batch_size=3,
        device=torch.device("cpu")
    )

    idx = torch.tensor([0, 2])
    sliced = cast(BatchBeam, beam[idx])
    assert torch.equal(cast(torch.Tensor, sliced.score), torch.tensor([1.0, 3.0]))
    assert sliced.state == "sliced_state"
    assert torch.equal(cast(torch.Tensor, sliced.parent), torch.tensor([0, 1]))
    assert torch.equal(cast(torch.Tensor, sliced.action), torch.tensor([1, 3]))


def test_batch_beam_clear_state():
    """Test clear_state method."""
    beam = BatchBeam(score=cast(Any, None), state="state", parent=None, action=None, batch_size=1, device=None)
    cleared = beam.clear_state()
    assert cleared.state is None
    assert cleared.score is None


def test_cached_lookup_mismatched_key_length():
    """Test CachedLookup when key length changes."""
    data = torch.tensor([10, 20, 30])
    lookup = CachedLookup(data)

    # First access
    val1 = cast(Any, lookup[torch.tensor([0, 1])])
    assert len(val1) == 2

    # Second access with different length (should recompute)
    val2 = cast(Any, lookup[torch.tensor([0, 1, 2])])
    assert len(val2) == 3


def test_cached_lookup_non_tensor():
    """Test CachedLookup correctly handles non-tensor keys after fix."""
    data = [10, 20, 30]
    lookup = CachedLookup(data)
    assert lookup[0] == 10
    assert lookup[1] == 20
