"""
Meta-Reinforcement Learning Trainer Implementations.

This module provides trainers that implement meta-learning strategies for dynamic
adaptation of cost weights and model parameters:
- RWATrainer: Reward Weight Adaptation using learnable meta-models.
- ContextualBanditTrainer: Dynamic weight selection using contextual bandits.
- TDLTrainer: Task-Dependent Learning with adaptive cost weights.
- MORLTrainer: Multi-Objective Reinforcement Learning.
- HyperNetworkTrainer: Meta-learning with hypernetworks.
- HRLTrainer: Hierarchical RL trainer with Manager-Worker architecture.
"""

import torch

from logic.src.models import GATLSTManager, HypernetworkOptimizer, WeightAdjustmentRNN
from logic.src.pipeline.reinforcement_learning.core.reinforce import (
    StandardTrainer,
    TimeTrainer,
)
from logic.src.pipeline.reinforcement_learning.meta.contextual_bandits import (
    WeightContextualBandit,
)
from logic.src.pipeline.reinforcement_learning.meta.multi_objective import (
    MORLWeightOptimizer,
)
from logic.src.pipeline.reinforcement_learning.meta.temporal_difference_learning import (
    CostWeightManager,
)
from logic.src.pipeline.reinforcement_learning.meta.weight_optimizer import (
    RewardWeightOptimizer,
)


class RWATrainer(StandardTrainer):
    """
    Reward Weight Adaptation (RWA) Trainer.

    Uses a meta-learning model (RNN or other) to dynamically adapt cost function weights
    during training. The meta-model observes training performance and adjusts weights to
    optimize for better solutions.

    The RewardWeightOptimizer learns to predict optimal weight configurations based on:
    - Historical performance metrics
    - Current cost distribution
    - Training dynamics

    This approach automates the hyperparameter tuning of cost weights, adapting them
    throughout training rather than keeping them fixed.
    """

    def __init__(
        self,
        model,
        optimizer,
        baseline,
        lr_scheduler,
        scaler,
        val_dataset,
        problem,
        tb_logger,
        cost_weights,
        opts,
    ):
        """
        Initialize the RWATrainer.
        """
        super().__init__(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            scaler,
            val_dataset,
            problem,
            tb_logger,
            cost_weights,
            opts,
        )
        self.weight_optimizer = RewardWeightOptimizer(num_objectives=len(cost_weights))

    def setup_meta_learner(self):
        """
        Initialize the Reward Weight Optimizer (RWO) meta-model.
        """
        model_class = None
        if self.opts["rwo_model"] == "rnn":
            model_class = WeightAdjustmentRNN

        len_weights = len(self.cost_weights.keys())
        min_weights = self.opts["meta_range"][0] * len_weights
        max_weights = self.opts["meta_range"][1] * len_weights

        self.weight_optimizer = RewardWeightOptimizer(
            model_class=model_class,
            initial_weights=self.cost_weights,
            history_length=self.opts.get("meta_history", 10),
            hidden_size=self.opts.get("mrl_embedding_dim", 64),
            lr=self.opts.get("mrl_lr", 0.001),
            device=self.opts["device"],
            meta_batch_size=self.opts.get("mrl_batch_size", 8),
            min_weights=min_weights,
            max_weights=max_weights,
            meta_optimizer=self.opts.get("rwa_optimizer", "adam"),
        )

    def update_context(self):
        """
        Update context hook (no-op for RWA).
        """
        pass


class ContextualBanditTrainer(TimeTrainer):
    """
    Contextual Bandit Trainer for dynamic weight selection.

    Uses a multi-armed bandit approach to select from a discrete set of weight configurations.
    The bandit algorithm (e.g., UCB, Thompson Sampling) learns which weight configurations
    work best in different contexts/situations.

    Unlike RWA which learns continuous weight adaptations, this trainer:
    - Maintains a fixed portfolio of weight configurations
    - Selects one configuration per day based on exploration-exploitation tradeoff
    - Updates selection probabilities based on observed rewards

    Combines with TimeTrainer to handle time-dependent training scenarios.
    """

    def __init__(
        self,
        model,
        optimizer,
        baseline,
        lr_scheduler,
        scaler,
        val_dataset,
        problem,
        tb_logger,
        cost_weights,
        opts,
    ):
        """
        Initialize the ContextualBanditTrainer.
        """
        super().__init__(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            scaler,
            val_dataset,
            problem,
            tb_logger,
            cost_weights,
            opts,
        )
        self.weight_optimizer = None

    def setup_meta_learner(self):
        """
        Initialize the Contextual Bandit weight optimizer.
        """
        self.weight_optimizer = WeightContextualBandit(
            num_weight_configs=10,
            initial_weights=self.cost_weights,
            # Epsilon and decay rate handled in update call or defaults
        )

    def update_context(self):
        """
        Update weights via Bandit proposal.
        """
        if self.weight_optimizer:
            weights = self.weight_optimizer.propose_weights(context=None)
            self.cost_weights.update(weights)

    def process_feedback(self):
        """
        Provide feedback (reward) to the Bandit optimizer.
        """
        if self.weight_optimizer:
            avg_cost = sum([torch.stack(c).mean().item() for c in self.log_costs]) / len(self.log_costs)
            reward = -avg_cost
            self.weight_optimizer.feedback(reward, metrics=None, day=self.day)


class TDLTrainer(StandardTrainer):
    """
    Temporal-Difference Learning (TDL) Trainer.

    Manages cost weights adaptively based on task characteristics using a CostWeightManager.
    The manager adjusts weights to balance multiple objectives dynamically during training.
    """

    def __init__(
        self,
        model,
        optimizer,
        baseline,
        lr_scheduler,
        scaler,
        val_dataset,
        problem,
        tb_logger,
        cost_weights,
        opts,
    ):
        """
        Initialize the TDLTrainer.
        """
        super().__init__(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            scaler,
            val_dataset,
            problem,
            tb_logger,
            cost_weights,
            opts,
        )
        self.weight_optimizer = CostWeightManager(num_objectives=len(cost_weights))


class MORLTrainer(StandardTrainer):
    """
    Multi-Objective Reinforcement Learning (MORL) Trainer.

    Optimizes for multiple objectives simultaneously using a MORLWeightOptimizer.
    Learns to find Pareto-optimal solutions that balance competing objectives
    (e.g., minimizing distance vs. minimizing overflows vs. maximizing waste collected).
    """

    def __init__(
        self,
        model,
        optimizer,
        baseline,
        lr_scheduler,
        scaler,
        val_dataset,
        problem,
        tb_logger,
        cost_weights,
        opts,
    ):
        """
        Initialize the MORLTrainer.
        """
        super().__init__(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            scaler,
            val_dataset,
            problem,
            tb_logger,
            cost_weights,
            opts,
        )
        self.weight_optimizer = MORLWeightOptimizer(num_objectives=len(cost_weights))


class HyperNetworkTrainer(TimeTrainer):
    """
    HyperNetwork Meta-Learning Trainer.

    Uses a hypernetwork to generate model parameters or weights based on task context.
    The hypernetwork learns to produce optimal configurations for different problem instances
    or environmental conditions, enabling fast adaptation to new scenarios.

    Combines meta-learning with time-dependent training for dynamic waste collection.
    """

    def __init__(
        self,
        model,
        optimizer,
        baseline,
        lr_scheduler,
        scaler,
        val_dataset,
        problem,
        tb_logger,
        cost_weights,
        opts,
    ):
        """
        Initialize the HyperNetworkTrainer.
        """
        super().__init__(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            scaler,
            val_dataset,
            problem,
            tb_logger,
            cost_weights,
            opts,
        )
        self.hyper_optimizer = HypernetworkOptimizer(
            cost_weight_keys=list(cost_weights.keys()),
            constraint_value=opts.get("constraint_value", 1.0),
            device=opts["device"],
            problem=problem,
        )


class HRLTrainer(StandardTrainer):
    """
    Hierarchical Reinforcement Learning (HRL) Trainer.

    Implements a two-level hierarchy:
    - Manager (GATLSTManager): Decides WHEN to dispatch collection vehicles
    - Worker (Attention Model): Decides WHICH route to take when dispatched

    The manager observes temporal bin fill patterns and outputs gating probabilities,
    while the worker focuses on solving the routing subproblem. This decomposition
    helps tackle the joint routing and scheduling challenge more effectively.
    """

    def __init__(
        self,
        model,
        optimizer,
        baseline,
        lr_scheduler,
        scaler,
        val_dataset,
        problem,
        tb_logger,
        cost_weights,
        opts,
    ):
        """
        Initialize the HRLTrainer.
        """
        super().__init__(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            scaler,
            val_dataset,
            problem,
            tb_logger,
            cost_weights,
            opts,
        )
        self.hrl_manager = GATLSTManager(device=opts["device"])
