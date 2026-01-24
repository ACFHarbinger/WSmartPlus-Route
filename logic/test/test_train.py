"""Tests for the modernized PyTorch Lightning training pipeline."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from logic.src.cli.train_lightning import create_model, run_training
from logic.src.configs import Config
from logic.src.pipeline.trainer import WSTrainer


class TestTrainingOrchestration:
    """Tests for train_lightning.py orchestration logic."""

    @pytest.mark.unit
    def test_create_model_reinforce(self):
        """Verify create_model creates a REINFORCE LightningModule."""
        cfg = Config()
        cfg.rl.algorithm = "reinforce"
        cfg.model.name = "am"

        with patch("logic.src.cli.train_lightning.get_env"), patch(
            "logic.src.cli.train_lightning.AttentionModelPolicy"
        ):
            model = create_model(cfg)
            from logic.src.pipeline.rl.core.reinforce import REINFORCE

            assert isinstance(model, REINFORCE)

    @pytest.mark.unit
    def test_create_model_ppo(self):
        """Verify create_model creates a PPO LightningModule."""
        cfg = Config()
        cfg.rl.algorithm = "ppo"
        cfg.model.name = "am"

        with patch("logic.src.cli.train_lightning.get_env"), patch(
            "logic.src.cli.train_lightning.AttentionModelPolicy"
        ), patch("logic.src.models.policies.critic.create_critic_from_actor"):
            model = create_model(cfg)
            from logic.src.pipeline.rl.core.ppo import PPO

            assert isinstance(model, PPO)

    @pytest.mark.unit
    @patch("logic.src.cli.train_lightning.WSTrainer")
    @patch("logic.src.cli.train_lightning.create_model")
    def test_run_training_flow(self, mock_create, mock_trainer_cls):
        """Verify run_training initializes model and calls trainer.fit."""
        cfg = Config()
        mock_model = MagicMock()
        mock_create.return_value = mock_model

        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.callback_metrics = {"val/reward": torch.tensor(0.5)}

        reward = run_training(cfg)

        assert reward == 0.5
        mock_create.assert_called_once_with(cfg)
        mock_trainer.fit.assert_called_once_with(mock_model)


class TestWSTrainer:
    """Tests for the custom WSTrainer class."""

    @pytest.mark.unit
    def test_wstrainer_init_defaults(self):
        """Verify WSTrainer sets up default callbacks and loggers."""
        with patch("logic.src.pipeline.trainer.WSTrainer._create_default_logger"):
            trainer = WSTrainer(max_epochs=5)
            assert trainer.max_epochs == 5
            # Check for ModelCheckpoint and RichProgressBar in callbacks
            callback_types = [type(c) for c in trainer.callbacks]
            from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

            assert ModelCheckpoint in callback_types
            assert RichProgressBar in callback_types

    @pytest.mark.unit
    def test_wstrainer_custom_logger(self):
        """Verify WSTrainer accepts a custom logger."""
        from pytorch_lightning.loggers import Logger

        custom_logger = MagicMock(spec=Logger)
        trainer = WSTrainer(logger=custom_logger)
        assert trainer.logger == custom_logger


class TestLitModuleInterfaces:
    """Tests for RL4COLitModule and algorithm implementations."""

    @pytest.mark.unit
    def test_reinforce_calculate_loss(self):
        """Simple test for REINFORCE loss calculation logic."""
        from tensordict import TensorDict

        from logic.src.pipeline.rl.core.reinforce import REINFORCE

        # Mock policy and baseline
        policy = MagicMock()
        env = MagicMock()

        model = REINFORCE(policy=policy, env=env)
        mock_baseline = MagicMock(spec=torch.nn.Module)
        model.baseline = mock_baseline
        model.baseline.eval.return_value = torch.tensor([1.0])

        td = TensorDict({}, batch_size=[1])
        out = {"reward": torch.tensor([2.0]), "log_likelihood": torch.tensor([-0.5], requires_grad=True)}

        loss = model.calculate_loss(td, out, 0, env=env)
        # advantage = 2 - 1 = 1.0 (after normalization might be different but here we check it runs)
        assert loss is not None
        assert loss.requires_grad
