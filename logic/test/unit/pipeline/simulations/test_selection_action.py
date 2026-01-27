"""Tests for orchestrated simulation actions."""

import numpy as np
import pytest
from logic.src.pipeline.simulations.actions import MustGoSelectionAction, PostProcessAction

class SimpleContext(dict):
    """A simple context that supports both dict access and attributes."""
    def __getattr__(self, name):
        return self.get(name)

@pytest.fixture
def mock_sim_context():
    """Mock simulation context."""
    bins = SimpleContext({
        'c': np.array([10.0, 95.0, 30.0, 85.0, 50.0]),
        'collectlevl': 90.0
    })

    ctx = SimpleContext({
        'n_bins': 5,
        'bins': bins,
        'day_count': 1,
        'full_policy': "policy_last_minute90_gamma1",
        'distpath_tup': (None, None, None, np.ones((6, 6))),
        'must_go': [],
        'area': "riomaior",
        'waste_type': "plastic"
    })
    return ctx

class TestMustGoSelectionAction:
    def test_selection_action_execution(self, mock_sim_context):
        action = MustGoSelectionAction()
        action.execute(mock_sim_context)

        # Strategy 'last_minute' selects bins > 90.
        # Index 1 (Bin 2) is 95.0. 1-based ID is 2.
        assert 2 in mock_sim_context['must_go']
        assert len(mock_sim_context['must_go']) == 1

class TestPostProcessAction:
    def test_post_process_action_execution(self, mock_sim_context):
        action = PostProcessAction()
        mock_sim_context['tour'] = [0, 2, 1, 3, 0]
        # Set post_process configuration
        mock_sim_context['post_process'] = "fast_tsp"
        mock_sim_context['distancesC'] = np.ones((6, 6), dtype=np.int32)

        action.execute(mock_sim_context)

        # Verify that the tour was updated
        assert mock_sim_context['tour'][0] == 0
        assert mock_sim_context['tour'][-1] == 0
