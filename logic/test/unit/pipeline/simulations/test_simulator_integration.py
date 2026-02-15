import os
import statistics
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock
from logic.src.pipeline.simulations import simulator
from logic.src.pipeline.simulations.bins import Bins
from logic.src.pipeline.simulations.checkpoints import CheckpointError

class TestSimulatorIntegration:
    """Integration tests for the WSmart+ Route simulator."""

    @pytest.mark.integration
    def test_single_simulation_happy_path_am(self, wsr_opts, mock_lock_counter, mock_torch_device, mocker, tmp_path):
        """Test single simulation with happy path."""
        opts = wsr_opts
        opts["policies"] = ["means_std_policy_am_gamma1"]
        opts["days"] = 5
        opts["size"] = 3
        N_BINS = 3

        mocker.patch("logic.src.policies.neural_agent.simulation.get_route_cost", return_value=50.0)
        mocker.patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path))
        mocker.patch("logic.src.pipeline.simulations.checkpoints.persistence.ROOT_DIR", str(tmp_path))

        # Mock data loading
        depot_df = pd.DataFrame({"ID": [0], "Lat": [40.0], "Lng": [-8.0], "Stock": [0], "Accum_Rate": [0]})
        bins_raw_df = pd.DataFrame({"ID": [1, 2, 3], "Lat": [40.1, 40.2, 40.3], "Lng": [-8.1, -8.2, -8.3]})
        data_raw_df = pd.DataFrame({"ID": [1, 2, 3], "Stock": [10, 20, 30], "Accum_Rate": [0, 0, 0]})
        mocker.patch("logic.src.pipeline.simulations.processor.setup_basedata", return_value=(data_raw_df, bins_raw_df, depot_df))
        mocker.patch("logic.src.pipeline.simulations.processor.setup_df", side_effect=[pd.DataFrame({"ID":[0,1,2,3]}), MagicMock()])
        mocker.patch("logic.src.pipeline.simulations.processor.process_data", side_effect=lambda data, bins_coords, depot, indices: (data, bins_coords))
        mocker.patch("logic.src.utils.data.data_utils.load_area_and_waste_type_params", return_value=(4000, 0.16, 60.0, 1.0, 2.5))

        mock_dist_tup = (np.zeros((4,4)), MagicMock(), torch.zeros((4,4)), np.zeros((4,4)))
        mocker.patch("logic.src.pipeline.simulations.processor.setup_dist_path_tup", return_value=(mock_dist_tup, np.zeros((4,4))))
        mocker.patch("logic.src.pipeline.simulations.processor.process_model_data", return_value=({"waste": MagicMock()}, (MagicMock(), MagicMock()), None))

        mock_model_env = MagicMock()
        mock_model_env.return_value = (MagicMock(), MagicMock(), {"waste": np.array([10.0])}, torch.tensor([0, 1, 2, 3, 0]), MagicMock())
        mock_model_env.temporal_horizon = 0
        mocker.patch("logic.src.pipeline.simulations.states.initializing.setup_model", return_value=(mock_model_env, MagicMock()))

        mocker.patch.object(Bins, "stochasticFilling", side_effect=lambda self, **k: (30.0, np.zeros(3), np.zeros(3), 0.0), autospec=True)
        mocker.patch.object(Bins, "setGammaDistribution", autospec=True)
        mocker.patch.object(Bins, "is_stochastic", return_value=True)
        mocker.patch.object(Bins, "get_fill_history", return_value=np.zeros((5, N_BINS)))

        simulator._lock, simulator._counter = mock_lock_counter
        result = simulator.single_simulation(opts, mock_torch_device, pol_id=0)
        assert result["success"]

    @pytest.mark.integration
    def test_single_simulation_resume(self, wsr_opts, mock_sim_dependencies, mock_lock_counter, mock_torch_device):
        """Test a simulation that resumes from a checkpoint."""
        opts = wsr_opts
        opts["resume"] = True
        mock_saved_state = ("d", "c", ("d", "p", "t", "c"), "a", mock_sim_dependencies["bins"], "m", None, 0, 0, {"day": [1,2,3,4,5]}, 0)
        mock_sim_dependencies["checkpoint"].load_state.return_value = (mock_saved_state, 5)

        simulator._lock, simulator._counter = mock_lock_counter
        result = simulator.single_simulation(opts, mock_torch_device, pol_id=0)
        assert result["success"]
        assert mock_sim_dependencies["run_day"].call_count == 5

    @pytest.mark.integration
    def test_single_simulation_checkpoint_error(self, wsr_opts, mocker, mock_lock_counter, mock_torch_device):
        """Test that CheckpointError is caught and returned."""
        mocker.patch("logic.src.pipeline.simulations.processor.setup_basedata", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
        mocker.patch("logic.src.pipeline.simulations.processor.setup_dist_path_tup", return_value=((None, None, None, None), None))
        mocker.patch("logic.src.pipeline.simulations.bins.Bins", return_value=MagicMock())
        error_result = {"success": False, "error": "test error"}
        mocker.patch("logic.src.pipeline.simulations.states.running.checkpoint_manager", side_effect=CheckpointError(error_result))

        simulator._lock, simulator._counter = mock_lock_counter
        result = simulator.single_simulation(wsr_opts, mock_torch_device, pol_id=0)
        assert result == error_result

    @pytest.mark.integration
    def test_sequential_simulations_multi_sample(self, wsr_opts, mock_sim_dependencies, mock_lock_counter, mock_torch_device):
        """Test sequential simulation with n_samples > 1."""
        opts = wsr_opts
        opts["n_samples"] = 2
        opts["days"] = 5
        opts["policies"] = ["policy_gamma1"]

        log, log_std, failed = simulator.sequential_simulations(opts, mock_torch_device, [None, None], [[0, 1]], lock=mock_lock_counter[0])
        assert mock_sim_dependencies["run_day"].call_count == 10
        assert "policy_gamma1" in log
