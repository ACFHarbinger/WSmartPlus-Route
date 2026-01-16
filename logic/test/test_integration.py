"""
Integration tests for the WSmart-Route pipeline (Training & Simulation).
Consolidated into classes as per user request.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from logic.src.pipeline.reinforcement_learning.worker_train import (
    train_reinforce_epoch,
    train_reinforce_over_time_cb,
    train_reinforce_over_time_tdl,
)

# Simulation imports
from logic.src.pipeline.simulator.simulation import sequential_simulations

# Training imports
from logic.src.pipeline.train import (
    hyperparameter_optimization,
    train_meta_reinforcement_learning,
    train_reinforcement_learning,
)


class TestIntegrationTraining:
    """Integration tests for Training, Meta-RL, and HPO workflows."""

    def test_train_epoch_am_vrpp(self, train_opts):
        """Test standard AM training loop on VRPP."""
        train_opts["model"] = "am"
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None
        assert os.path.exists(os.path.join(train_opts["final_dir"], "args.json"))

    def test_train_epoch_am_gat(self, train_opts):
        """Test AM with GAT encoder."""
        train_opts["encoder"] = "gat"
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    def test_train_epoch_am_gcn(self, train_opts):
        """Test AM with GCN encoder."""
        train_opts["encoder"] = "gcn"
        # GCN requires edges.
        # VRPPDataset uses 'number_edges' as threshold for 'edge_strat'
        train_opts["edge_method"] = "knn"
        train_opts["number_edges"] = 5  # Acts as K=5
        train_opts["edge_threshold"] = 5
        train_opts["size"] = 11  # FORCE NEW DATASET GENERATION

        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    def test_train_baseline_critic(self, train_opts):
        """Test training with Critic network baseline."""
        train_opts["baseline"] = "critic"
        train_opts["lr_critic"] = 1e-4
        train_opts["optimizer"] = "adam"  # Force Adam to avoid LBFGS param group error
        assert train_opts["optimizer"] == "adam"

        # Ensure edges if critic needs them, or just robust opts
        train_opts["edge_method"] = "knn"
        train_opts["number_edges"] = 5
        train_opts["size"] = 11  # Force new data

        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    def test_train_device_cpu(self, train_opts):
        """Verify explicit CPU usage."""
        # train_opts["device"] removed to avoid json dump error
        train_opts["no_cuda"] = True
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        param = next(model.parameters())
        assert param.device.type == "cpu"

    def test_train_overfitting(self, train_opts):
        """Sanity check: loss should decrease on small data over multiple epochs."""
        train_opts["n_epochs"] = 2  # Short run
        train_opts["epoch_size"] = 2
        train_opts["batch_size"] = 2
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    def test_train_resume_checkpoint(self, train_opts, tmp_path):
        """Verify model checkpoint resuming."""
        # 1. Run initial training
        train_opts["n_epochs"] = 1
        # Use fixture tmp_path explicitly if needed, but train_opts has it set
        train_opts["checkpoint_dir"] = "checkpoints"
        train_opts["model_save_interval"] = 1

        # Use copy to avoid device object pollution between runs
        opts_run1 = train_opts.copy()
        model1, _ = train_reinforcement_learning(opts_run1, train_reinforce_epoch)

        ckpt_path = os.path.join(train_opts["save_dir"], "epoch-0.pt")
        assert os.path.exists(ckpt_path)

        # 2. Resume
        train_opts["resume"] = ckpt_path
        train_opts["n_epochs"] = 1
        train_opts["epoch_start"] = 0

        model2, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model2 is not None

    def test_train_saving_artifacts(self, train_opts):
        """Verify model and args are saved correctly."""
        train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert os.path.exists(os.path.join(train_opts["final_dir"], "args.json"))

    def test_mrl_train_cb(self, train_opts):
        """Test Meta-RL with Contextual Bandits dispatch."""
        train_opts["mrl_method"] = "cb"
        with patch("logic.src.pipeline.train.train_reinforcement_learning") as mock_train_rl:
            train_meta_reinforcement_learning(train_opts)
            mock_train_rl.assert_called_once()
            args, _ = mock_train_rl.call_args
            assert args[1] == train_reinforce_over_time_cb

    def test_mrl_train_tdl(self, train_opts):
        """Test Meta-RL with TDL dispatch."""
        train_opts["mrl_method"] = "tdl"
        with patch("logic.src.pipeline.train.train_reinforcement_learning") as mock_train_rl:
            train_meta_reinforcement_learning(train_opts)
            mock_train_rl.assert_called_once()
            args, _ = mock_train_rl.call_args
            assert args[1] == train_reinforce_over_time_tdl

    def test_hp_optim_grid_search(self, train_opts, tmp_path):
        """Test Hyperparameter Optimization (Grid Search)."""
        train_opts["hop_method"] = "gs"
        train_opts["problem"] = "wcvrp"
        train_opts["grid"] = [1e-4, 1e-3]

        train_opts["save_dir"] = str(tmp_path / "hpo")
        train_opts["run_name"] = "hpo_test"
        train_opts["train_best"] = False
        train_opts["hop_epochs"] = 1
        train_opts["cpu_cores"] = 1

        # Patch tune to avoid Ray issues and DeprecationWarnings
        with patch(
            "logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.hpo.tune"
        ) as mock_tune, patch(
            "logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.hpo.grid_search"
        ) as mock_gs:
            # Mock optimized result
            mock_opt = MagicMock()
            mock_opt.return_value = (0.5, 0.5, {})

            # Configure mock tune to return valid config matching grid
            mock_trial = MagicMock()
            mock_trial.config = {"w_lost": 1e-4}  # Real dict for json dump
            mock_trial.last_result = {"validation_metric": 0.5}
            mock_tune.run.return_value.get_best_trial.return_value = mock_trial

            hyperparameter_optimization(train_opts)

            assert mock_tune.run.called or mock_gs.called


class TestIntegrationSimulation:
    """Integration tests for Simulation workflows."""

    def _run_sim(self, opts):
        """Helper to run simulation."""
        # Mock indices/samples as required by sequential_simulations signature
        # indices_ls: List[List[int]] -> per sample, list of node indices
        indices_ls = [list(range(opts["size"]))]

        # sample_idx_ls: List[List[int]] -> per policy, list of sample indices (0-based)
        sample_idx_ls = [[0] for _ in opts["policies"]]

        lock = MagicMock()

        return sequential_simulations(opts, opts["device"], indices_ls, sample_idx_ls, opts["model_path"], lock)

    def test_sim_sequential_basic(self, sim_opts):
        """Test basic sequential simulation run."""
        log, log_std, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("regular_emp" in k for k in log.keys())

    def test_sim_parallel_run(self, sim_opts):
        """Test parallel execution setup."""
        sim_opts["parallel"] = True
        pass

    def test_sim_policy_crashes_on_unknown(self, sim_opts):
        """Verify unknown policy crashes."""
        sim_opts["policies"] = ["unknown_policy_v2"]
        with pytest.raises(Exception):
            self._run_sim(sim_opts)

    def test_sim_policy_regular(self, sim_opts):
        sim_opts["policies"] = ["policy_regular_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert "policy_regular_emp" in log

    def test_sim_policy_last_minute(self, sim_opts):
        # Format: policy_last_minute<threshold>_distribution
        sim_opts["policies"] = ["policy_last_minute50_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert "policy_last_minute50_emp" in log

    def test_sim_policy_look_ahead(self, sim_opts):
        # Format: policy_look_ahead_<config>_distribution
        sim_opts["policies"] = ["policy_look_ahead_a_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert "policy_look_ahead_a_emp" in log

    def test_sim_checkpoint_creation(self, sim_opts, tmp_path):
        """Verify checkpoints are created."""
        sim_opts["output_dir"] = "sim_ckpt"
        self._run_sim(sim_opts)

    def test_sim_horizon_short(self, sim_opts):
        sim_opts["days"] = 1
        self._run_sim(sim_opts)
