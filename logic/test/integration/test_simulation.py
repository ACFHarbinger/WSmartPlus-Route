from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.pipeline.simulations.day_context import run_day
from logic.src.pipeline.simulations.simulator import single_simulation
from logic.src.pipeline.simulations.states import InitializingState, SimulationContext


@pytest.fixture
def mock_sim_opts(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return {
        "area": "Rio Maior",
        "size": 10,
        "n_samples": 1,
        "seed": 42,
        "days": 2,
        "policies": ["regular_unif"],  # Needs distribution suffix usually
        "verbosity": 0,
        "output_dir": "test_sim_out",
        "data_dir": str(data_dir),
        "resume": False,
        "store": False,
        "no_render": True,
        "save_video": False,
        "capacity": 100,
        "waste_type": "glass",
        "n_vehicles": 1,
        "distance_method": "osrm",
        "dm_filepath": None,
        "env_file": None,
        "gapik_file": None,
        "symkey_name": None,
        "edge_threshold": 0.0,
        "edge_method": "knn",
        "vertex_method": "geo",
        "model_path": None,
        "temperature": 1.0,
        "strategy": "greedy",
        "checkpoint_days": 1,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "garbage_filepath": None,
        "stats_filepath": None,
        "waste_filepath": None,
        "no_progress_bar": True,
        "data_distribution": "unif",  # Explicitly set
    }


@pytest.mark.integration
def test_simulation_context_init(mock_sim_opts):
    """Test initializing SimulationContext and transitioning to InitializingState."""
    device = torch.device("cpu")
    indices = [0, 1, 2]

    with patch("logic.src.pipeline.simulations.states.SimulationContext.transition_to") as mock_transition:
        context = SimulationContext(
            opts=mock_sim_opts,
            device=device,
            indices=indices,
            sample_id=0,
            pol_id=0,
            model_weights_path=None,
            variables_dict={},
        )

        # Should initialize and transition to InitializingState
        assert context.opts == mock_sim_opts
        assert context.policy == "regular_unif"
        mock_transition.assert_called()
        # Verify call args
        args, _ = mock_transition.call_args
        assert isinstance(args[0], InitializingState)


@pytest.mark.integration
def test_run_day_execution():
    """Test run_day function orchestrates actions."""
    # Mock SimulationDayContext
    mock_context = MagicMock()
    mock_context.policy = "regular"

    with patch("logic.src.pipeline.simulations.actions.FillAction") as MockFill, \
         patch("logic.src.pipeline.simulations.actions.PolicyExecutionAction") as MockPolicy, \
         patch("logic.src.pipeline.simulations.actions.CollectAction") as MockCollect, \
         patch("logic.src.pipeline.simulations.actions.LogAction") as MockLog, \
         patch("logic.src.pipeline.simulations.actions.MustGoSelectionAction") as MockSelection, \
         patch("logic.src.pipeline.simulations.actions.PostProcessAction") as MockPostProcess:

        run_day(mock_context)

        MockFill.return_value.execute.assert_called_with(mock_context)
        MockSelection.return_value.execute.assert_called_with(mock_context)
        MockPolicy.return_value.execute.assert_called_with(mock_context)
        MockPostProcess.return_value.execute.assert_called_with(mock_context)
        MockCollect.return_value.execute.assert_called_with(mock_context)
        MockLog.return_value.execute.assert_called_with(mock_context)


@pytest.mark.integration
def test_single_simulation_wrapper(mock_sim_opts):
    """Test the single_simulation wrapper function."""
    with patch("logic.src.pipeline.simulations.simulator.SimulationContext") as MockContext:
        instance = MockContext.return_value
        instance.run.return_value = {"success": True, "regular_unif": {"profit": 100}}

        result = single_simulation(
            opts=mock_sim_opts,
            device=torch.device("cpu"),
            indices=[0],
            sample_id=0,
            pol_id=0,
            model_weights_path=None,
            n_cores=1,
        )

        assert result["success"] is True
        assert result["regular_unif"]["profit"] == 100
