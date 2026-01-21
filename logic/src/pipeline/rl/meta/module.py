"""
Meta-Reinforcement Learning (Meta-RL) module.

This module implements bi-level optimization where a meta-learner
adjusts environment reward weights to optimize the training progress of an inner RL agent.
"""
from typing import Any

import pytorch_lightning as pl
import torch
from tensordict import TensorDict

from logic.src.models.meta_rnn import WeightAdjustmentRNN
from logic.src.pipeline.rl.meta import get_meta_strategy


class MetaRLModule(pl.LightningModule):
    """
    Lightning module for Meta-Reinforcement Learning.

    Wraps an RL agent (inner loop) and a meta-model (outer loop).
    """

    def __init__(
        self,
        agent: Any,
        strategy: str = "rnn",
        meta_lr: float = 1e-3,
        history_length: int = 10,
        hidden_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["agent"])
        self.agent = agent

        # Initial weights configuration
        initial_weights = {
            "collection": 10.0,  # Default for WCVRPP
            "cost": 1.0,
        }

        # Strategy selection
        strategy_kwargs = {
            "model_class": WeightAdjustmentRNN,
            "initial_weights": initial_weights,
            "history_length": history_length,
            "hidden_size": hidden_size,
            "lr": meta_lr,
            "device": str(self.device),
            "meta_optimizer": "adam",
            **kwargs,
        }

        self.meta_strategy = get_meta_strategy(strategy, **strategy_kwargs)

        # Disable automatic optimization for the meta-learner to control outer loop updates
        self.automatic_optimization = False

    def training_step(self, batch: TensorDict, batch_idx: int):
        """Step for Meta-RL bi-level optimization."""
        # 1. Inner Loop: Train the Agent
        loss = self.agent.training_step(batch, batch_idx)

        # 2. Outer Loop: Update Meta-Learner
        if hasattr(self.agent, "last_out"):
            out = self.agent.last_out
            reward = out["reward"].mean().item()

            # Pack metrics for strategy
            metrics = [
                out.get("collection", torch.tensor(0.0)).mean().item(),
                out.get("cost", torch.tensor(0.0)).mean().item(),
            ]

            # Feedback to meta-strategy
            self.meta_strategy.feedback(reward, metrics)

            # 3. Adapt weights for next step
            new_weights = self.meta_strategy.propose_weights()

            # Update environment attributes (Duck Typing)
            for name, val in new_weights.items():
                attr_name = f"{name}_reward" if name == "collection" else f"{name}_weight"
                if hasattr(self.agent.env, attr_name):
                    setattr(self.agent.env, attr_name, val)

                self.log(f"meta/weight_{name}", val)

        return loss

    def configure_optimizers(self):
        """Configure optimizers for both loops."""
        return self.agent.configure_optimizers()

    def validation_step(self, batch: TensorDict, batch_idx: int):
        return self.agent.validation_step(batch, batch_idx)

    def test_step(self, batch: TensorDict, batch_idx: int):
        return self.agent.test_step(batch, batch_idx)
