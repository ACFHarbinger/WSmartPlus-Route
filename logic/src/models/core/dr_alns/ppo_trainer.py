"""PPO Trainer for DR-ALNS optimization.

This module implements the training logic for the DR-ALNS agent, including
the experience replay buffer (`PPOBuffer`) and the `PPOTrainer` which
manages the environment interactions, advantage estimation, and gradient
updates using the PPO algorithm.

Attributes:
    PPOBuffer: Container for on-policy experience collection.
    PPOTrainer: Coordinator for agent-environment optimization loops.

Example:
    >>> from logic.src.models.core.dr_alns.ppo_trainer import PPOTrainer
    >>> trainer = PPOTrainer(agent, env)
    >>> history = trainer.train(total_timesteps=10000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

if TYPE_CHECKING:
    from logic.src.envs.dr_alns import DRALNSEnv

from .ppo_agent import DRALNSPPOAgent


class PPOBuffer:
    """Experience storage for Proximal Policy Optimization.

    Maintains trajectories of states, actions, log-probabilities, rewards,
    done flags, and value estimates collected during environmental rollouts.

    Attributes:
        capacity (int): Maximum steps to store.
        states (List[torch.Tensor]): History of observation vectors [7].
        actions (List[Dict[str, int]]): History of discrete operator indices.
        log_probs (List[Dict[str, torch.Tensor]]): History of policy probabilities.
        rewards (List[float]): History of immediate collection rewards.
        dones (List[bool]): History of episode termination flags.
        values (List[torch.Tensor]): History of critic value estimates.
    """

    def __init__(self, capacity: int = 10000) -> None:
        """Initializes the experience buffer.

        Args:
            capacity: Buffer size limit.
        """
        self.capacity = capacity
        self.reset()

    def reset(self) -> None:
        """Flushes all stored trajectories to prepare for new rollouts."""
        self.states: List[torch.Tensor] = []
        self.actions: List[Dict[str, int]] = []
        self.log_probs: List[Dict[str, torch.Tensor]] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def add(
        self,
        state: torch.Tensor,
        action: Dict[str, int],
        log_prob: Dict[str, torch.Tensor],
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        """Appends a new transition to the storage.

        Args:
            state: Current observation.
            action: Selected discrete actions map.
            log_prob: Log-probabilities of selected actions.
            reward: Scalar reward from step.
            done: Whether episode terminated.
            value: Estimated state value.
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self, device: torch.device) -> Dict[str, Any]:
        """Batches all stored data into tensors for training.

        Args:
            device: hardware target for tensors.

        Returns:
            Dict[str, Any]: Dictionary of batched experience tensors.
        """
        states = torch.stack(self.states).to(device)

        # Batch discrete operator indices
        actions_dict = {
            "destroy": torch.tensor([a["destroy"] for a in self.actions], device=device, dtype=torch.long),
            "repair": torch.tensor([a["repair"] for a in self.actions], device=device, dtype=torch.long),
            "severity": torch.tensor([a["severity"] for a in self.actions], device=device, dtype=torch.long),
            "temp": torch.tensor([a["temp"] for a in self.actions], device=device, dtype=torch.long),
        }

        # Batch policy log-probs
        log_probs_dict = {
            "destroy": torch.stack([lp["destroy"] for lp in self.log_probs]).to(device),
            "repair": torch.stack([lp["repair"] for lp in self.log_probs]).to(device),
            "severity": torch.stack([lp["severity"] for lp in self.log_probs]).to(device),
            "temp": torch.stack([lp["temp"] for lp in self.log_probs]).to(device),
        }

        rewards = torch.tensor(self.rewards, device=device, dtype=torch.float32)
        dones = torch.tensor(self.dones, device=device, dtype=torch.float32)
        values = torch.stack(self.values).to(device).squeeze(-1)

        return {
            "states": states,
            "actions": actions_dict,
            "log_probs": log_probs_dict,
            "rewards": rewards,
            "dones": dones,
            "values": values,
        }

    def __len__(self) -> int:
        """Returns the current step count in the buffer.

        Returns:
            int: Number of stored experience steps.
        """
        return len(self.states)


class PPOTrainer:
    """Orchestrator for DR-ALNS neural agent optimization.

    Implements the PPO pipeline: collection of on-policy rollouts, computation of
    GAE advantages, and stochastic gradient ascent updates.

    Attributes:
        agent (DRALNSPPOAgent): The multi-head model being trained.
        env (DRALNSEnv): Gymnasium interface to the ALNS solver.
        gamma (float): Future reward discount factor.
        gae_lambda (float): Smoothing parameter for advantage estimation.
        clip_epsilon (float): PPO conservation range.
        value_loss_coef (float): Critic regularization weight.
        entropy_coef (float): Exploration regularization weight.
        max_grad_norm (float): Gradient clipping threshold.
        n_epochs (int): Number of SGD passes per collection batch.
        batch_size (int): Minibatch size for gradient steps.
        device (torch.device): Working hardware.
        optimizer (Adam): Neural network optimizer.
        buffer (PPOBuffer): Experience storage.
    """

    def __init__(
        self,
        agent: DRALNSPPOAgent,
        env: DRALNSEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes the PPO trainer.

        Args:
            agent: The PPO agent.
            env: The target ALNS environment.
            lr: Learning rate.
            gamma: Discount factor.
            gae_lambda: GAE parameter.
            clip_epsilon: PPO clip ratio.
            value_loss_coef: Critic weighting.
            entropy_coef: Exploration weighting.
            max_grad_norm: Gradient norm limit.
            n_epochs: SGD update repeat count.
            batch_size: Training batch size.
            device: Compute device.
        """
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        self.agent.to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=lr)

        self.buffer = PPOBuffer()

    def collect_experience(
        self, n_steps: int, instance_generator: Optional[Callable[..., Any]] = None
    ) -> Dict[str, float]:
        """Interacts with the ALNS environment to harvest state-action-reward tuples.

        Args:
            n_steps: Number of total environment steps to harvest.
            instance_generator: logic for problem reset if applicable.

        Returns:
            Dict[str, float]: Rollout statistics (mean reward, best profit, etc).
        """
        self.agent.eval()
        self.buffer.reset()

        episode_rewards = []
        episode_best_profits = []

        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_best_profit = info.get("best_profit", 0.0)

        for _ in range(n_steps):
            state_tensor = torch.from_numpy(obs).float().to(self.device)

            with torch.no_grad():
                actions, log_probs, value = self.agent.get_action(state_tensor)

            next_obs, reward, terminated, truncated, info = self.env.step(
                np.array(
                    [
                        actions["destroy"],
                        actions["repair"],
                        actions["severity"],
                        actions["temp"],
                    ]
                )
            )

            done = terminated or truncated
            episode_reward += reward
            episode_best_profit = max(episode_best_profit, info.get("best_profit", 0.0))

            self.buffer.add(state_tensor, actions, log_probs, reward, done, value)

            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_best_profits.append(episode_best_profit)

                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_best_profit = info.get("best_profit", 0.0)

        stats = {
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_best_profit": float(np.mean(episode_best_profits)) if episode_best_profits else 0.0,
            "n_episodes": float(len(episode_rewards)),
        }

        return stats

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates advantages using the GAE algorithm.

        Args:
            rewards: Trajectory rewards [N].
            values: trajectory critic estimates [N].
            dones: Trajectory termination flags [N].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Advantage and Return tensors.
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae: torch.Tensor | float = 0.0
        next_value: torch.Tensor | float = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Performs PPO gradient updates based on buffered experience.

        Returns:
            Dict[str, float]: Optimization metrics (policy loss, value loss, entropy).
        """
        self.agent.train()

        batch = self.buffer.get(self.device)
        advantages, returns = self.compute_advantages(batch["rewards"], batch["values"], batch["dones"])

        # Advantage normalization stabilizes updates
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        n_samples = len(batch["states"])

        for _ in range(self.n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]

                mb_states = batch["states"][mb_indices]
                actions_batch = cast(Dict[str, torch.Tensor], batch["actions"])
                log_probs_batch = cast(Dict[str, torch.Tensor], batch["log_probs"])
                mb_actions = {k: v[mb_indices] for k, v in actions_batch.items()}
                mb_old_log_probs = {k: v[mb_indices] for k, v in log_probs_batch.items()}
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Actor-critic evaluation
                new_log_probs, new_values, entropies = self.agent.evaluate_actions(mb_states, mb_actions)

                # Compute head-averaged importance ratios
                ratios = {}
                for key in new_log_probs:
                    ratios[key] = torch.exp(new_log_probs[key] - mb_old_log_probs[key])

                avg_ratio = torch.stack(list(ratios.values())).mean(dim=0)

                # PPO Clipped Objective
                surr1 = avg_ratio * mb_advantages
                surr2 = torch.clamp(avg_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # MSE Value loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # Head-averaged entropy bonus
                avg_entropy = torch.stack(list(entropies.values())).mean()

                # Unified loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * avg_entropy

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += avg_entropy.item()

        n_updates = self.n_epochs * (n_samples // self.batch_size)
        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

        return stats

    def train(
        self,
        total_timesteps: int,
        n_steps_per_update: int = 2048,
        log_interval: int = 10,
        instance_generator: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, List[float]]:
        """Conducts a full training run over multiple rollout/update cycles.

        Args:
            total_timesteps: total budget of interactions.
            n_steps_per_update: Batch size of history to collect before an update.
            log_interval: updates between metrics printing.
            instance_generator: Environment problem generation logic.

        Returns:
            Dict[str, List[float]]: Metric histories.
        """
        history: Dict[str, List[float]] = {
            "mean_episode_reward": [],
            "mean_best_profit": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

        n_updates = total_timesteps // n_steps_per_update

        for update in range(n_updates):
            collect_stats = self.collect_experience(n_steps_per_update, instance_generator)
            update_stats = self.update()

            if (update + 1) % log_interval == 0:
                print(
                    f"Update {update + 1}/{n_updates} | "
                    f"Reward: {collect_stats['mean_episode_reward']:.2f} | "
                    f"Best Profit: {collect_stats['mean_best_profit']:.2f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                    f"Value Loss: {update_stats['value_loss']:.4f}"
                )

            history["mean_episode_reward"].append(collect_stats["mean_episode_reward"])
            history["mean_best_profit"].append(collect_stats["mean_best_profit"])
            history["policy_loss"].append(update_stats["policy_loss"])
            history["value_loss"].append(update_stats["value_loss"])
            history["entropy"].append(update_stats["entropy"])

        return history
