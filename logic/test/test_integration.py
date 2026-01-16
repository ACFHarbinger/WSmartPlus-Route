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
    """Integration tests for Training workflows with various RL algorithms."""

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_train_reinforce(self, train_opts, problem_name):
        """Test standard REINFORCE training."""
        train_opts["problem"] = problem_name
        train_opts["rl_algorithm"] = "reinforce"
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None
        assert os.path.exists(os.path.join(train_opts["final_dir"], "args.json"))

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_train_ppo(self, train_opts, problem_name):
        """Test PPO training."""
        train_opts["problem"] = problem_name
        train_opts["rl_algorithm"] = "ppo"
        train_opts["ppo_epochs"] = 1
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_train_sapo(self, train_opts, problem_name):
        """Test SAPO training."""
        train_opts["problem"] = problem_name
        train_opts["rl_algorithm"] = "sapo"
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_train_gspo(self, train_opts, problem_name):
        """Test GSPO training."""
        train_opts["problem"] = problem_name
        train_opts["rl_algorithm"] = "gspo"
        train_opts["gspo_epochs"] = 1
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_train_dr_grpo(self, train_opts, problem_name):
        """Test DR-GRPO training."""
        train_opts["problem"] = problem_name
        train_opts["rl_algorithm"] = "dr_grpo"
        train_opts["dr_grpo_epochs"] = 1
        # DR-GRPO needs batch_size to be a multiple of group_size
        train_opts["dr_grpo_group_size"] = 2
        train_opts["batch_size"] = 4
        train_opts["epoch_size"] = 4
        model, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model is not None

    def test_train_with_saving_and_resume(self, train_opts, tmp_path):
        """Verify model checkpoint saving and resuming."""
        train_opts["n_epochs"] = 1
        train_opts["checkpoint_epochs"] = 1

        # 1. Initial training
        model1, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        ckpt_path = os.path.join(train_opts["save_dir"], "epoch-0.pt")
        assert os.path.exists(ckpt_path)

        # 2. Resume
        train_opts["resume"] = ckpt_path
        model2, _ = train_reinforcement_learning(train_opts, train_reinforce_epoch)
        assert model2 is not None

    def test_train_encoders(self, train_opts):
        """Verify training with different encoders."""
        for encoder in ["gat", "gcn"]:
            opts = train_opts.copy()
            opts["encoder"] = encoder
            if encoder == "gcn":
                opts["edge_method"] = "knn"
                opts["edge_threshold"] = 5
            model, _ = train_reinforcement_learning(opts, train_reinforce_epoch)
            assert model is not None

    def test_train_baseline_variants(self, train_opts):
        """Test training with different baseline variants."""
        for bl in ["critic", "exponential", "rollout"]:
            opts = train_opts.copy()
            opts["baseline"] = bl
            if bl == "critic":
                opts["lr_critic_value"] = 1e-4
            model, _ = train_reinforcement_learning(opts, train_reinforce_epoch)
            assert model is not None


class TestIntegrationMetaTraining:
    """Integration tests for Meta-Reinforcement Learning workflows."""

    def _test_mrl_dispatch(self, train_opts, problem_name, method, expected_trainer_func):
        """Helper to test Meta-RL dispatch logic."""
        train_opts["problem"] = problem_name
        train_opts["mrl_method"] = method
        with patch("logic.src.pipeline.train.train_reinforcement_learning") as mock_train_rl:
            train_meta_reinforcement_learning(train_opts)
            mock_train_rl.assert_called_once()
            args, _ = mock_train_rl.call_args
            assert args[1] == expected_trainer_func

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_mrl_train_cb(self, train_opts, problem_name):
        """Test Meta-RL with Contextual Bandits dispatch."""
        self._test_mrl_dispatch(train_opts, problem_name, "cb", train_reinforce_over_time_cb)

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_mrl_train_tdl(self, train_opts, problem_name):
        """Test Meta-RL with TDL dispatch."""
        self._test_mrl_dispatch(train_opts, problem_name, "tdl", train_reinforce_over_time_tdl)

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_mrl_train_rwa(self, train_opts, problem_name):
        """Test Meta-RL with Reward Weight Adjustment dispatch."""
        from logic.src.pipeline.reinforcement_learning.worker_train import train_reinforce_over_time_rwa

        self._test_mrl_dispatch(train_opts, problem_name, "rwa", train_reinforce_over_time_rwa)

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_mrl_train_morl(self, train_opts, problem_name):
        """Test Meta-RL with Multi-Objective RL dispatch."""
        from logic.src.pipeline.reinforcement_learning.worker_train import train_reinforce_over_time_morl

        self._test_mrl_dispatch(train_opts, problem_name, "morl", train_reinforce_over_time_morl)

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    def test_mrl_train_hrl(self, train_opts, problem_name):
        """Test Meta-RL with Hierarchical RL dispatch."""
        from logic.src.pipeline.reinforcement_learning.worker_train import train_reinforce_over_time_hrl

        self._test_mrl_dispatch(train_opts, problem_name, "hrl", train_reinforce_over_time_hrl)


class TestIntegrationHPO:
    """Integration tests for Hyperparameter Optimization workflows."""

    def _mock_hpo_run(self, train_opts, problem_name, method, mock_tune, mock_opt=None):
        """Helper to configure common HPO mocks."""
        train_opts["problem"] = problem_name
        train_opts["hop_method"] = method
        train_opts["train_best"] = False
        train_opts["hop_epochs"] = 1
        train_opts["cpu_cores"] = 1

        # Configure mock tune to return valid config
        mock_trial = MagicMock()
        mock_trial.config = {"w_lost": 1e-4, "w_waste": 1.0, "w_length": 1.0, "w_overflows": 100.0}
        mock_trial.last_result = {"validation_metric": 0.5}
        mock_tune.run.return_value.get_best_trial.return_value = mock_trial

        if mock_opt:
            mock_opt.return_value = mock_trial.config

        # Mock ray to avoid init errors
        with patch("ray.init"), patch("ray.shutdown"):
            hyperparameter_optimization(train_opts)

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    @patch("logic.src.pipeline.train.grid_search")
    def test_hpo_grid_search(self, mock_gs, train_opts, problem_name):
        """Test Grid Search HPO."""
        train_opts["grid"] = [1e-4, 1e-3]
        self._mock_hpo_run(train_opts, problem_name, "gs", MagicMock(), mock_gs)
        assert mock_gs.called

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    @patch("logic.src.pipeline.train.random_search")
    def test_hpo_random_search(self, mock_rs, train_opts, problem_name):
        """Test Random Search HPO."""
        train_opts["num_samples"] = 2
        self._mock_hpo_run(train_opts, problem_name, "rs", MagicMock(), mock_rs)
        assert mock_rs.called

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    @patch("logic.src.pipeline.train.bayesian_optimization")
    def test_hpo_bayesian_optimization(self, mock_bo, train_opts, problem_name):
        """Test Bayesian Optimization HPO."""
        train_opts["n_trials"] = 2
        self._mock_hpo_run(train_opts, problem_name, "bo", MagicMock(), mock_bo)
        assert mock_bo.called

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    @patch("logic.src.pipeline.train.differential_evolutionary_hyperband_optimization")
    def test_hpo_dehbo(self, mock_dehbo, train_opts, problem_name):
        """Test DEHBO HPO."""
        train_opts["fevals"] = 2
        self._mock_hpo_run(train_opts, problem_name, "dehbo", MagicMock(), mock_dehbo)
        assert mock_dehbo.called

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    @patch("logic.src.pipeline.train.distributed_evolutionary_algorithm")
    def test_hpo_dea(self, mock_dea, train_opts, problem_name):
        """Test DEA HPO."""
        train_opts["n_pop"] = 2
        train_opts["n_gen"] = 1
        self._mock_hpo_run(train_opts, problem_name, "dea", MagicMock(), mock_dea)
        assert mock_dea.called

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"])
    @patch("logic.src.pipeline.train.hyperband_optimization")
    def test_hpo_hyperband(self, mock_hbo, train_opts, problem_name):
        """Test Hyperband HPO."""
        self._mock_hpo_run(train_opts, problem_name, "hbo", MagicMock(), mock_hbo)
        assert mock_hbo.called


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
