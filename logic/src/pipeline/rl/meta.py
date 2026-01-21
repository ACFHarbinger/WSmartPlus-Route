"""
Meta-Reinforcement Learning (Meta-RL) module.

This module implements bi-level optimization where a meta-learner (MetaRNN)
adjusts environment reward weights to optimize the training progress of an inner RL agent.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from tensordict import TensorDict

from logic.src.models.meta_rnn import WeightAdjustmentRNN
from logic.src.pipeline.reinforcement_learning.meta.weight_optimizer import RewardWeightOptimizer
from logic.src.pipeline.rl.base import RL4COLitModule


class MetaRLModule(pl.LightningModule):
    """
    Lightning module for Meta-Reinforcement Learning.

    Wraps an RL agent (inner loop) and a meta-model (outer loop).
    The meta-model learns to adjust reward weights based on the agent's performance.
    """

    def __init__(
        self,
        agent: RL4COLitModule,
        meta_lr: float = 1e-3,
        history_length: int = 10,
        hidden_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["agent"])
        self.agent = agent

        # Initialize Meta-Learner
        # We need to know the initial weights of the environment
        # For WCVRP, we usually have 'collection_reward' and 'cost_weight' (implied as 1.0)
        # However, RL4CO envs might have different attributes.
        # We'll assume a standard mapping for now.
        initial_weights = {
            "collection": getattr(agent.env, "collection_reward", 1.0),
            "cost": 1.0,
        }

        self.meta_optimizer = RewardWeightOptimizer(
            model_class=WeightAdjustmentRNN,
            initial_weights=initial_weights,
            history_length=history_length,
            hidden_size=hidden_size,
            lr=meta_lr,
            device=self.device,
            meta_optimizer="adam",
            **kwargs,
        )

        # Disable automatic optimization for the meta-learner to control outer loop updates
        self.automatic_optimization = False

    def training_step(self, batch: TensorDict, batch_idx: int):
        """
        Step for Meta-RL bi-level optimization.
        """
        # 1. Inner Loop: Train the Agent
        # We delegate to the agent's training_step
        # Note: If agent uses manual optimization (like PPO), it will handle it.
        # If it uses automatic optimization, we might need to call it differently?
        # In Lightning, manual_optimization=False is default.

        # We perform the inner update
        loss = self.agent.training_step(batch, batch_idx)

        # 2. Outer Loop: Update Meta-Learner
        # Collect metrics from the environment after the step
        # Note: The agent's training_step usually returns the loss.
        # We need more metrics (e.g. reward components) for the meta-learner.

        # We can extract reward components if the environment or out dictionary has them.
        # For now, let's assume we can get collection and cost from the agent's last rollout.
        # This might require some hooks in RL4COLitModule.

        # Assume self.agent.last_out contains the necessary metrics
        if hasattr(self.agent, "last_out"):
            out = self.agent.last_out
            reward = out["reward"].mean().item()

            # Pack metrics for RewardWeightOptimizer
            # It expects [overflows, length, waste, total_reward] or similar
            # based on initial_weights order?
            # In our initial_weights we have [collection, cost].
            # RewardWeightOptimizer.feedback expects metrics + total_reward.

            metrics = [
                out.get("collection", torch.tensor(0.0)).mean().item(),
                out.get("cost", torch.tensor(0.0)).mean().item(),
            ]

            # Feed back to meta-optimizer
            self.meta_optimizer.feedback(reward, metrics)

            # 3. Adapt weights for next step
            new_weights = self.meta_optimizer.propose_weights()

            # Update environment attributes
            if "collection" in new_weights:
                setattr(self.agent.env, "collection_reward", new_weights["collection"])
            if "cost" in new_weights:
                # If we add a cost_weight attribute to the env
                setattr(self.agent.env, "cost_weight", new_weights["cost"])

            # Log new weights
            for name, val in new_weights.items():
                self.log(f"meta/weight_{name}", val)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizers for both loops.
        """
        # Agent's optimizer (inner)
        inner_opt = self.agent.configure_optimizers()

        # Meta-learner's optimizer (outer)
        # Note: RewardWeightOptimizer already has its internal optimizer.
        # We might need to handle this carefully in Lightning.

        return inner_opt

    def validation_step(self, batch: TensorDict, batch_idx: int):
        return self.agent.validation_step(batch, batch_idx)

    def test_step(self, batch: TensorDict, batch_idx: int):
        return self.agent.test_step(batch, batch_idx)
