"""
PPO algorithm implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.pipeline.rl.base import RL4COLitModule


class PPO(RL4COLitModule):
    """
    Proximal Policy Optimization (PPO).

    Implements PPO for constructive routing problems.
    Uses a Critic network as a value baseline and performs multiple epochs
    of optimization per batch of rollouts.
    """

    def __init__(
        self,
        critic: nn.Module,
        ppo_epochs: int = 10,
        eps_clip: float = 0.2,
        value_loss_weight: float = 0.5,
        entropy_weight: float = 0.0,
        max_grad_norm: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.critic = critic
        self.ppo_epochs = ppo_epochs
        self.eps_clip = eps_clip
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm

        # We use manual optimization to allow multiple epochs per batch
        self.automatic_optimization = False

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Dummy implementation to satisfy abstract requirement.
        PPO uses manual optimization in training_step.
        """
        return torch.tensor(0.0, device=td.device)

    def training_step(self, batch: TensorDict, batch_idx: int):
        env = self.env
        td = env.reset(batch.clone())

        with torch.no_grad():
            # Sampling rollout
            out = self.policy(td, env, decode_type="sampling")
            old_actions = out["actions"]  # (batch, steps)
            old_log_p = out["log_likelihood"]  # (batch)
            old_reward = out["reward"]  # (batch)

            # Initial state value
            self.critic(batch).squeeze(-1)  # (batch)

        # 2. PPO Optimization Loop
        opt = self.optimizers()

        for _ in range(self.ppo_epochs):
            # Re-evaluate log probabilities of the same actions
            # Teacher forcing using 'actions' argument
            td_new = env.reset(batch.clone())
            new_out = self.policy(td_new, env, actions=old_actions)
            new_log_p = new_out["log_likelihood"]

            # Re-evaluate values
            new_values = self.critic(batch).squeeze(-1)

            # Advantage estimation
            advantage = self.calculate_advantages(old_reward, new_values)

            # Calculate Ratio
            ratio = self.calculate_ratio(new_log_p, old_log_p)

            # Actor Loss
            actor_loss = self.calculate_actor_loss(ratio, advantage)

            # Critic Loss
            critic_loss = self.calculate_critic_loss(new_values, old_reward)

            # Total Loss
            loss = actor_loss + self.value_loss_weight * critic_loss

            # Entropy bias (optional)
            if self.entropy_weight > 0 and "entropy" in new_out:
                loss = loss - self.entropy_weight * new_out["entropy"].mean()

            # Manual Optimization
            opt.zero_grad()
            self.manual_backward(loss)

            if self.max_grad_norm > 0:
                # Clip gradients for both policy and critic
                self.clip_gradients(opt, gradient_clip_val=self.max_grad_norm, gradient_clip_algorithm="norm")

            opt.step()

        # Log metrics (using values from last optimization step)
        self.log("train/reward", old_reward.mean(), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/actor_loss", actor_loss)
        self.log("train/critic_loss", critic_loss)
        self.log("train/advantage", advantage.mean())
        self.log("train/ratio", ratio.mean())

        return loss

    def calculate_advantages(self, rewards, values):
        """Estimate advantages (R - V)."""
        advantage = rewards - values.detach()
        if advantage.size(0) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return advantage

    def calculate_ratio(self, new_log_p, old_log_p):
        """Calculate importance ratio."""
        return torch.exp(new_log_p - old_log_p.detach())

    def calculate_actor_loss(self, ratio, advantage):
        """Clipped surrogate objective."""
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
        return -torch.min(surr1, surr2).mean()

    def calculate_critic_loss(self, values, rewards):
        """MSE loss for value function."""
        return nn.MSELoss()(values, rewards)

    def configure_optimizers(self):
        # Combined parameters from policy and critic
        params = list(self.policy.parameters()) + list(self.critic.parameters())
        return torch.optim.Adam(params, **self.optimizer_kwargs)
