"""
Integration tests for the WSmart-Route pipeline (Training & Simulation).
Modernized for the PyTorch Lightning architecture.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

# Training imports from the new entry point
from logic.src.cli.train_lightning import run_training
from logic.src.configs import Config

# Simulation imports
from logic.src.pipeline.simulations.simulator import sequential_simulations


class TestIntegrationTraining:
    """Integration tests for Training workflows."""

    @pytest.mark.parametrize("problem_name", ["vrpp", "wcvrp"])
    def test_run_training_integration(self, problem_name, tmp_path):
        """Test standard training orchestration with real entry point (mocked trainer)."""
        cfg = Config()
        cfg.env.name = problem_name
        cfg.train.n_epochs = 1
        cfg.train.logs_dir = str(tmp_path / "logs")
        cfg.train.final_model_path = str(tmp_path / "final.pt")

        # We mock WSTrainer to avoid actual training in unit tests,
        # but check if run_training flow completes.
        with patch("logic.src.cli.train_lightning.WSTrainer") as mock_trainer_cls, patch(
            "logic.src.cli.train_lightning.create_model"
        ):
            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.callback_metrics = {"val/reward": torch.tensor(0.1)}

            reward = run_training(cfg)
            assert isinstance(reward, float)
            mock_trainer.fit.assert_called_once()


class TestIntegrationSimulation:
    """Integration tests for Simulation workflows."""

    def _run_sim(self, opts):
        """Helper to run simulation."""
        # Mock indices/samples as required by sequential_simulations signature
        indices_ls = [list(range(opts["size"]))]
        sample_idx_ls = [[0] for _ in opts["policies"]]
        lock = MagicMock()

        return sequential_simulations(opts, opts["device"], indices_ls, sample_idx_ls, opts["model_path"], lock)

    @pytest.mark.unit
    def test_sim_sequential_basic(self, sim_opts):
        """Test basic sequential simulation run."""
        # Ensure we have some policies to test
        sim_opts["policies"] = ["policy_regular_emp"]
        log, log_std, failed = self._run_sim(sim_opts)
        assert not failed
        assert any("regular_emp" in k for k in log.keys())

    @pytest.mark.unit
    @patch("logic.src.pipeline.simulations.states.setup_model")
    @patch("logic.src.policies.neural_agent.NeuralPolicy.execute")
    def test_sim_policy_neural_mock(self, mock_exec, mock_setup, sim_opts):
        """Test Neural policy integration in simulation."""
        mock_setup.return_value = (MagicMock(), {})
        mock_exec.return_value = ([0, 1, 2, 0], 10.0, None)
        sim_opts["policies"] = ["am_emp"]
        log, _, failed = self._run_sim(sim_opts)
        assert not failed
        assert "am_emp" in log


class TestIntegrationProblems:
    """Tests for problem physics and state transitions (End-to-End)."""

    def test_vrpp_physics_flow(self):
        """Verify VRPP physics behaves correctly in an end-to-end state update."""
        from logic.src.envs.problems import VRPP

        batch = {
            "loc": torch.rand(1, 10, 2),
            "depot": torch.rand(1, 2),
            "waste": torch.rand(1, 10),
            "max_waste": torch.ones(1, 10),
        }
        state = VRPP.make_state(batch)
        assert state.td["i"] == 0

        # Take an action
        action = torch.tensor([1])
        next_state = state.update(action)
        assert next_state.td["i"] == 1
        assert next_state.td["visited"][:, 1].all()

    def test_wcvrp_physics_flow(self):
        """Verify WCVRP physics behaves correctly."""
        from logic.src.envs.problems import WCVRP

        batch = {
            "loc": torch.rand(1, 5, 2),
            "depot": torch.rand(1, 2),
            "waste": torch.rand(1, 5),
            "max_waste": torch.ones(1, 5),
        }
        edges = torch.ones(1, 5, 5)
        state = WCVRP.make_state(batch, edges=edges)

        action = torch.tensor([1])
        next_state = state.update(action)
        assert next_state.td["i"] == 1
