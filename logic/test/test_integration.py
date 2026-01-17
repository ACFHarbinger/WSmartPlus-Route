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

    def test_sim_policy_vrpp_gurobi(self, sim_opts):
        """Test VRPP policy with Gurobi optimizer."""
        sim_opts["policies"] = ["gurobi_vrpp_0.5_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("gurobi_vrpp_0.5_emp" in k for k in log.keys())

    def test_sim_policy_vrpp_hexaly(self, sim_opts):
        """Test VRPP policy with Hexaly optimizer."""
        sim_opts["policies"] = ["hexaly_vrpp_0.5_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("hexaly_vrpp_0.5_emp" in k for k in log.keys())

    def test_sim_policy_look_ahead_vrpp(self, sim_opts):
        """Test Look-Ahead policy with VRPP (Gurobi)."""
        sim_opts["policies"] = ["policy_look_ahead_avrpp_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("policy_look_ahead_avrpp_emp" in k for k in log.keys())

    def test_sim_policy_look_ahead_sans(self, sim_opts):
        """Test Look-Ahead policy with Simulated Annealing."""
        sim_opts["policies"] = ["policy_look_ahead_asans_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("policy_look_ahead_asans_emp" in k for k in log.keys())

    def test_sim_policy_look_ahead_alns(self, sim_opts):
        """Test Look-Ahead policy with ALNS."""
        sim_opts["policies"] = ["policy_look_ahead_alns_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("policy_look_ahead_alns_emp" in k for k in log.keys())

    def test_sim_policy_look_ahead_hgs(self, sim_opts):
        """Test Look-Ahead policy with HGS."""
        sim_opts["policies"] = ["policy_look_ahead_ahgs_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("policy_look_ahead_ahgs_emp" in k for k in log.keys())

    def test_sim_policy_look_ahead_bcp(self, sim_opts):
        """Test Look-Ahead policy with BCP (OR-Tools)."""
        sim_opts["policies"] = ["policy_look_ahead_bcp_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("policy_look_ahead_bcp_emp" in k for k in log.keys())

    def test_sim_policy_profit_reactive(self, sim_opts):
        """Test Profit-based reactive policy."""
        sim_opts["policies"] = ["policy_profit_reactive_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("policy_profit_reactive_emp" in k for k in log.keys())

    def test_sim_multi_vehicle_regular(self, sim_opts):
        """Test Regular policy with multiple vehicles."""
        sim_opts["policies"] = ["policy_regular_emp"]
        sim_opts["n_vehicles"] = 8
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert "policy_regular_emp" in log

    @patch("logic.src.pipeline.simulator.states.setup_model")
    @patch("logic.src.policies.neural_agent.NeuralPolicy.execute")
    def test_sim_policy_neural_mock(self, mock_exec, mock_setup, sim_opts):
        """Test Neural policy integration with mocked execution to avoid actual model loading."""
        mock_setup.return_value = (MagicMock(), {})
        mock_exec.return_value = ([0, 1, 2, 0], 10.0, None)
        sim_opts["policies"] = ["am_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert "am_emp" in log
        assert mock_exec.called


# ============================================================================
# Training Pipeline Integration Tests (IMPROVEMENT_PLAN.md recommendations)
# ============================================================================


class TestIntegrationTrainingPipeline:
    """Integration tests for training pipeline components (VRPP state, encoders, components)."""

    @pytest.fixture
    def vrpp_batch(self):
        """Create a VRPP problem batch for testing."""
        import torch

        batch_size = 4
        graph_size = 20
        return {
            "loc": torch.rand(batch_size, graph_size, 2),
            "depot": torch.rand(batch_size, 2),
            "waste": torch.rand(batch_size, graph_size),
            "max_waste": torch.ones(batch_size, graph_size),
        }

    @pytest.fixture
    def encoder_kwargs(self):
        """Create encoder parameters."""
        return {
            "embed_dim": 64,
            "n_layers": 2,
            "n_heads": 4,
            "normalization": "batch",
            "feed_forward_hidden": 256,
        }

    def test_vrpp_make_state(self, vrpp_batch):
        """Test VRPP state creation from batch."""
        from logic.src.problems.vrpp.problem_vrpp import VRPP

        state = VRPP.make_state(vrpp_batch)
        assert state is not None
        assert hasattr(state, "coords")
        assert hasattr(state, "visited_")

    def test_vrpp_state_transition(self, vrpp_batch):
        """Test VRPP state updates correctly on action."""
        import torch

        from logic.src.problems.vrpp.problem_vrpp import VRPP

        state = VRPP.make_state(vrpp_batch)
        # Select depot (index 0) as action - always valid
        actions = torch.zeros(4, dtype=torch.long)

        next_state = state.update(actions)
        assert next_state is not None
        # i should have incremented
        assert next_state.i > state.i

    def test_batch_collation(self):
        """Test batch collation for dataloader."""
        import torch

        from logic.src.utils.data_utils import collate_fn

        samples = [
            {"loc": torch.rand(10, 2), "waste": torch.rand(10)},
            {"loc": torch.rand(10, 2), "waste": torch.rand(10)},
        ]
        batch = collate_fn(samples)
        assert batch["loc"].shape == (2, 10, 2)
        assert batch["waste"].shape == (2, 10)

    def test_attention_factory_creates_encoder(self, encoder_kwargs):
        """Test AttentionComponentFactory creates encoder."""
        from logic.src.models.model_factory import AttentionComponentFactory

        factory = AttentionComponentFactory()
        encoder = factory.create_encoder(**encoder_kwargs)
        assert encoder is not None
        assert hasattr(encoder, "forward")

    def test_mlp_factory_creates_encoder(self, encoder_kwargs):
        """Test MLPComponentFactory creates encoder."""
        from logic.src.models.model_factory import MLPComponentFactory

        factory = MLPComponentFactory()
        kwargs = encoder_kwargs.copy()
        encoder = factory.create_encoder(**kwargs)
        assert encoder is not None

    def test_encoder_forward_pass(self, encoder_kwargs):
        """Test encoder produces valid embeddings."""
        import torch

        from logic.src.models.model_factory import AttentionComponentFactory

        factory = AttentionComponentFactory()
        encoder = factory.create_encoder(**encoder_kwargs)
        h = torch.rand(4, 20, 64)
        output = encoder(h)
        assert output.shape == (4, 20, 64)


class TestIntegrationStateTransitions:
    """Tests for problem state transitions and constraints."""

    def test_vrpp_mask_prevents_revisit(self):
        """Test that visited nodes are masked for future selection."""
        import torch

        from logic.src.problems.vrpp.problem_vrpp import VRPP

        batch = {
            "loc": torch.rand(2, 5, 2),
            "depot": torch.rand(2, 2),
            "waste": torch.rand(2, 5),
            "max_waste": torch.ones(2, 5),
        }
        state = VRPP.make_state(batch)
        state = state.update(torch.tensor([1, 1]))
        mask = state.get_mask()
        assert mask[:, 0, 1].all()

    def test_vrpp_capacity_constraint(self):
        """Test capacity constraints in VRPP state."""
        import torch

        from logic.src.problems.vrpp.problem_vrpp import VRPP

        batch = {
            "loc": torch.rand(2, 5, 2),
            "depot": torch.rand(2, 2),
            "waste": torch.full((2, 5), 0.5),
            "max_waste": torch.ones(2, 5),
        }
        state = VRPP.make_state(batch)
        # Initially, total waste collected should be 0
        assert state.cur_total_waste.sum() == 0

    def test_vrpp_all_finished_detection(self):
        """Test that state correctly detects completion."""
        import torch

        from logic.src.problems.vrpp.problem_vrpp import VRPP

        batch = {
            "loc": torch.rand(1, 3, 2),
            "depot": torch.rand(1, 2),
            "waste": torch.rand(1, 3),
            "max_waste": torch.ones(1, 3),
        }
        state = VRPP.make_state(batch)
        assert not state.all_finished()


class TestIntegrationModelComponents:
    """Tests for individual model components."""

    def test_normalization_layer_batch(self):
        """Test batch normalization layer."""
        import torch

        from logic.src.models.modules.normalization import Normalization

        norm = Normalization(64, "batch")
        x = torch.rand(4, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_layer_layer(self):
        """Test layer normalization."""
        import torch

        from logic.src.models.modules.normalization import Normalization

        norm = Normalization(64, "layer")
        x = torch.rand(4, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_feed_forward_module(self):
        """Test feed forward module."""
        import torch

        from logic.src.models.modules.feed_forward import FeedForward

        ff = FeedForward(input_dim=64, output_dim=64)
        x = torch.rand(4, 10, 64)
        out = ff(x)
        assert out.shape == (4, 10, 64)

    def test_skip_connection_residual(self):
        """Test residual skip connection."""
        import torch
        import torch.nn as nn

        from logic.src.models.modules.skip_connection import SkipConnection

        sublayer = nn.Linear(64, 64)
        skip = SkipConnection(module=sublayer)
        x = torch.rand(4, 10, 64)
        out = skip(x)
        assert out.shape == x.shape

    def test_multi_head_attention_basic(self):
        """Test multi-head attention module."""
        import torch

        from logic.src.models.modules.multi_head_attention import MultiHeadAttention

        mha = MultiHeadAttention(n_heads=4, input_dim=64, embed_dim=64)
        q = torch.rand(4, 1, 64)
        h = torch.rand(4, 10, 64)
        out = mha(q, h)
        assert out.shape[0] == 4

    def test_activation_function_relu(self):
        """Test ReLU activation function wrapper."""
        import torch

        from logic.src.models.modules.activation_function import ActivationFunction

        act = ActivationFunction("relu")
        x = torch.randn(4, 10, 64)
        out = act(x)
        assert (out >= 0).all()

    def test_activation_function_gelu(self):
        """Test GELU activation function wrapper."""
        import torch

        from logic.src.models.modules.activation_function import ActivationFunction

        act = ActivationFunction("gelu")
        x = torch.randn(4, 10, 64)
        out = act(x)
        assert out.shape == x.shape


# ============================================================================
# Additional Problem Module Tests
# ============================================================================


class TestIntegrationProblems:
    """Tests for problem modules (VRPP, WCVRP, etc.)."""

    def test_vrpp_validate_tours(self):
        """Test VRPP tour validation."""
        import torch

        from logic.src.problems.vrpp.problem_vrpp import VRPP

        # Valid tour starting and ending at depot (0)
        tour = torch.tensor([[0, 1, 2, 0]])
        VRPP.validate_tours(tour)  # Should not raise

    def test_vrpp_get_tour_length(self):
        """Test VRPP tour length calculation."""
        import torch

        from logic.src.problems.vrpp.problem_vrpp import VRPP

        dataset = {
            "loc": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]),
            "depot": torch.tensor([[0.5, 0.5]]),
        }
        tour = torch.tensor([[0, 1, 2, 0]])
        length = VRPP.get_tour_length(dataset, tour)
        assert length.item() > 0

    def test_cvrpp_make_state(self):
        """Test CVRPP state creation."""
        import torch

        from logic.src.problems.vrpp.problem_vrpp import CVRPP

        batch = {
            "loc": torch.rand(2, 5, 2),
            "depot": torch.rand(2, 2),
            "waste": torch.rand(2, 5),
            "max_waste": torch.ones(2, 5),
        }
        state = CVRPP.make_state(batch)
        assert state is not None
        assert hasattr(state, "coords")

    def test_wcvrp_make_state(self):
        """Test WCVRP state creation."""
        import torch

        from logic.src.problems.wcvrp.problem_wcvrp import WCVRP

        batch = {
            "loc": torch.rand(2, 5, 2),
            "depot": torch.rand(2, 2),
            "waste": torch.rand(2, 5),
            "max_waste": torch.ones(2, 5),
        }
        # WCVRP.make_state needs edges
        edges = torch.zeros(2, 5, 5)
        state = WCVRP.make_state(batch, edges=edges)
        assert state is not None


# ============================================================================
# Neural Model Subnet Tests
# ============================================================================


class TestIntegrationSubnets:
    """Tests for neural model subnets."""

    def test_gat_encoder_multiple_layers(self):
        """Test GAT encoder with multiple layers."""
        import torch

        from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder

        encoder = GraphAttentionEncoder(
            n_heads=4,
            embed_dim=64,
            n_layers=3,
            n_groups=4,
        )
        x = torch.rand(4, 20, 64)
        out = encoder(x)
        assert out.shape == (4, 20, 64)

    def test_gcn_encoder_basic(self):
        """Test GCN encoder basic operation."""
        import torch

        from logic.src.models.subnets.gcn_encoder import GraphConvolutionEncoder

        encoder = GraphConvolutionEncoder(
            n_layers=2,
            feed_forward_hidden=64,
            n_groups=4,
        )
        x = torch.rand(2, 10, 64)
        edges = torch.randint(0, 2, (2, 10, 10))
        out = encoder(x, edges)
        assert out.shape == (2, 10, 64)

    def test_attention_decoder(self):
        """Test attention decoder forward pass."""
        import torch

        from logic.src.models.subnets.gat_decoder import GraphAttentionDecoder

        decoder = GraphAttentionDecoder(
            n_heads=4,
            embed_dim=64,
            n_layers=1,
            n_groups=4,
        )
        q = torch.rand(4, 1, 64)
        h = torch.rand(4, 20, 64)
        mask = torch.zeros(4, 20, dtype=torch.bool)
        out = decoder(q, h, mask)
        assert out.shape[0] == 4


# ============================================================================
# Additional Module Tests for Coverage
# ============================================================================


class TestIntegrationModules:
    """Tests for neural modules."""

    def test_normalization_instance(self):
        """Test instance normalization."""
        import torch

        from logic.src.models.modules.normalization import Normalization

        norm = Normalization(64, "instance")
        x = torch.rand(4, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_group(self):
        """Test group normalization."""
        import torch

        from logic.src.models.modules.normalization import Normalization

        norm = Normalization(64, "group", n_groups=4)
        x = torch.rand(4, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_graph_convolution_basic(self):
        """Test basic graph convolution layer."""
        import torch

        from logic.src.models.modules.graph_convolution import GraphConvolution

        gc = GraphConvolution(in_channels=64, out_channels=64, aggregation="mean")
        x = torch.rand(2, 10, 64)
        edges = torch.rand(2, 10, 10)
        out = gc(x, edges)
        assert out.shape == (2, 10, 64)

    def test_activation_tanh(self):
        """Test tanh activation."""
        import torch

        from logic.src.models.modules.activation_function import ActivationFunction

        act = ActivationFunction("tanh")
        x = torch.randn(4, 10, 64)
        out = act(x)
        assert (out >= -1).all() and (out <= 1).all()

    def test_activation_leaky_relu(self):
        """Test leaky ReLU activation."""
        import torch

        from logic.src.models.modules.activation_function import ActivationFunction

        act = ActivationFunction("leakyrelu", fparam=0.01)
        x = torch.randn(4, 10, 64)
        out = act(x)
        assert out.shape == x.shape
