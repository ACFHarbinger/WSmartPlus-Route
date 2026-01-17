"""
Baseline Methods for Variance Reduction in REINFORCE.

This module implements various baseline strategies to reduce the variance of policy gradient
estimators in REINFORCE-style training. Baselines subtract a state-dependent value from the
reward to center the advantage, improving gradient estimation without introducing bias.

Available Baselines:
- NoBaseline: No variance reduction (baseline = 0)
- ExponentialBaseline: Exponential moving average of returns
- POMOBaseline: POMO (Policy Optimization with Multiple Optima) averaging
- CriticBaseline: Learned value function via critic network
- RolloutBaseline: Self-critic using greedy policy rollouts
- WarmupBaseline: Gradual transition between two baselines

Mathematical Foundation:
The policy gradient estimator is: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * (R - b(s))]
where b(s) is a baseline that depends only on state, not action.
A good baseline reduces variance while maintaining unbiasedness.

Reference: Attention, Learn to Solve Routing Problems (Kool et al., 2019)
"""

import copy

import scipy.stats as stats
import torch
import torch.nn.functional as F

from logic.src.pipeline.reinforcement_learning.core.epoch import (
    get_inner_model,
    rollout,
)


# Attention, Learn to Solve Routing Problems
class Baseline(object):
    """
    Abstract base class for all baseline methods.

    Baselines provide a state-dependent value estimate b(s) that is subtracted from
    rewards to compute advantages: A = R - b(s). This reduces gradient variance
    without introducing bias (since E[b(s)] doesn't depend on actions).

    Subclasses must implement the eval() method and can override other methods
    to customize behavior (dataset wrapping, parameter learning, etc.).
    """

    def wrap_dataset(self, dataset):
        """
        Optionally wrap the dataset to attach precomputed baseline values.

        Args:
            dataset: Training dataset

        Returns:
            Wrapped or original dataset
        """
        return dataset

    def unwrap_batch(self, batch):
        """
        Extract data and baseline values from a batch.

        Args:
            batch: Batch from dataloader (potentially wrapped)

        Returns:
            tuple: (batch_data, baseline_values)
        """
        return batch, None

    def eval(self, x, c):
        """
        Evaluate the baseline for a given state and cost.

        Args:
            x: State/input batch
            c: Cost/reward for the batch

        Returns:
            tuple: (baseline_value, baseline_loss)
                - baseline_value: Estimated baseline b(x)
                - baseline_loss: Loss for training the baseline (if learnable)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        """
        Get parameters that should be optimized during training.

        Returns:
            list: List of PyTorch parameters (empty for non-learnable baselines)
        """
        return []

    def epoch_callback(self, model, epoch):
        """
        Callback executed at the end of each training epoch.

        Allows baselines to update their state (e.g., RolloutBaseline updates
        its model if the current policy improves).

        Args:
            model: Current policy model
            epoch: Current epoch number
        """
        pass

    def state_dict(self):
        """
        Get baseline state for checkpointing.

        Returns:
            dict: State dictionary for saving
        """
        return {}

    def load_state_dict(self, state_dict):
        """
        Load baseline state from checkpoint.

        Args:
            state_dict: Previously saved state dictionary
        """
        pass


class WarmupBaseline(Baseline):
    """
    Gradual transition from a simple baseline to a more complex one.

    Useful for stabilizing training at the start when a RolloutBaseline or CriticBaseline
    might be unreliable. Starts with an ExponentialBaseline and gradually transitions to
    the target baseline over n_epochs using linear interpolation.

    The baseline value is computed as: b(x) = α * b_target(x) + (1 - α) * b_warmup(x)
    where α increases from 0 to 1 over the warmup period.

    Args:
        baseline: Target baseline to transition to
        n_epochs: Number of epochs for the warmup period
        warmup_exp_beta: Beta parameter for the ExponentialBaseline used during warmup
    """

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        """
        Initialize the WarmupBaseline.

        Args:
            baseline: The target baseline to transition to.
            n_epochs (int): Number of epochs for the warmup period.
            warmup_exp_beta (float): Beta for the exponential baseline.
        """
        super(Baseline, self).__init__()
        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        """
        Wrap dataset using current alpha-weighted baseline logic.
        """
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        """
        Unwrap batch using current alpha-weighted baseline logic.
        """
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        """
        Evaluate the warmup baseline (interpolation).
        """
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, loss = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * loss + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        """
        Update alpha based on current epoch.
        """
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        """
        Return state dict of inner baseline.
        """
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load state dict of inner baseline.
        """
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):
    """
    No baseline - returns zero for all states.

    This is the simplest baseline (no variance reduction). Useful for:
    - Debugging/ablation studies
    - Problems with naturally low variance
    - Sanity checks

    Results in the raw REINFORCE gradient estimator with high variance.
    """

    def eval(self, x, c):
        """Returns zero baseline and zero loss."""
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):
    """
    Exponential moving average (EMA) of returns as baseline.

    Maintains a running average: v_t = β * v_{t-1} + (1 - β) * R_t
    where β ∈ [0, 1] controls the smoothing (higher β = more smoothing).

    Pros:
    - Simple and fast
    - No additional parameters to learn
    - Works well for stationary reward distributions

    Cons:
    - Not state-dependent (same baseline for all states)
    - Can be slow to adapt to non-stationary rewards
    - Less effective than learned baselines

    Args:
        beta: Smoothing parameter (default: 0.8). Higher values = more smoothing.
    """

    def __init__(self, beta):
        """
        Initialize the ExponentialBaseline.

        Args:
            beta (float): Smoothing factor (0.8 means 80% old value, 20% new value).
        """
        super(Baseline, self).__init__()
        self.beta = beta
        self.v = None

    def eval(self, x, c):
        """
        Update and return the exponential moving average.

        Args:
            x: State (unused, baseline is state-independent)
            c: Current costs/rewards

        Returns:
            tuple: (baseline_value, 0) - no loss since non-learnable
        """
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        """
        Return the state dict (current average value).
        """
        return {"v": self.v}

    def load_state_dict(self, state_dict):
        """
        Load the state dict.
        """
        self.v = state_dict["v"]


class POMOBaseline(Baseline):
    """
    POMO (Policy Optimization with Multiple Optima) Baseline.

    Uses the average cost across multiple augmentations of the same problem instance
    as the baseline. For each problem instance, POMO generates multiple solutions
    (e.g., starting from different depot rotations) and uses their mean as the baseline.

    This provides a strong, instance-specific baseline without requiring additional
    model evaluations, since the augmentations are generated in parallel during training.

    Key idea: For instance i with K augmentations:
    - Costs: [c_i1, c_i2, ..., c_iK]
    - Baseline for each: b_ik = mean([c_i1, c_i2, ..., c_iK])

    Reference: POMO paper (Kwon et al., 2020)

    Args:
        pomo_size: Number of augmentations per instance (K)
    """

    def __init__(self, pomo_size):
        """
        Initialize the POMOBaseline.

        Args:
            pomo_size (int): Number of augmentations/rotations per instance.
        """
        super(Baseline, self).__init__()
        self.pomo_size = pomo_size

    def eval(self, x, c):
        """
        Evaluate POMO baseline (mean reward across augmentations).
        """
        # c: [batch_size * pomo_size]
        B_pomo = c.size(0)
        B = B_pomo // self.pomo_size

        # Reshape to [B, pomo_size]
        # rewards = c.reshape(B, self.pomo_size)
        rewards = c.view(B, self.pomo_size)

        # Compute mean reward per instance: [B]
        mean_rewards = rewards.mean(dim=1)

        # Repeat mean rewards to match c shape: [B * pomo_size]
        # Repeat mean rewards to match c shape: [B * pomo_size]
        v = mean_rewards.repeat_interleave(self.pomo_size)

        return v, 0  # No critic loss


class CriticBaseline(Baseline):
    """
    Learned value function baseline using a critic network.

    Trains a separate neural network (critic) to estimate V(s), the expected return
    from state s. The critic is trained via supervised learning to predict actual
    returns, minimizing MSE: L = (V(s) - R)²

    This is similar to the Actor-Critic architecture, but here the critic is only
    used as a baseline (not for policy improvement).

    Pros:
    - State-dependent baseline (more effective variance reduction)
    - Can capture complex value patterns
    - Commonly used in modern RL

    Cons:
    - Requires additional network and training
    - Can be unstable if critic training diverges
    - Adds computational overhead

    Args:
        critic: Neural network that maps states to value estimates
    """

    def __init__(self, critic):
        """
        Initialize the CriticBaseline.

        Args:
            critic: The critic network module.
        """
        super(Baseline, self).__init__()
        self.critic = critic

    def eval(self, x, c):
        """
        Evaluate critic and compute training loss.

        Args:
            x: State batch
            c: Actual costs/returns

        Returns:
            tuple: (value_estimate, mse_loss)
                - value_estimate: V(x) detached (for baseline)
                - mse_loss: MSE between V(x) and actual returns (for training)
        """
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v.squeeze(-1), c.detach())

    def get_learnable_parameters(self):
        """
        Get learnable parameters of the critic.
        """
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        """
        Callback at end of epoch (no-op for CriticBaseline).
        """
        pass

    def state_dict(self):
        """
        Return state dict of the critic.
        """
        return {"critic": self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        """
        Load state dict into the critic.
        """
        critic_state_dict = state_dict.get("critic", {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):
    """
    Self-critical baseline using greedy rollouts of the current policy.

    Maintains a snapshot of the policy from a previous epoch and uses its greedy
    solutions as the baseline. This is a form of "self-play" where the policy
    competes against its past self.

    Algorithm:
    1. Periodically save a copy of the current policy
    2. Generate greedy solutions using the saved policy for baseline values
    3. Train current policy using these solutions as baseline
    4. Update saved policy if current policy significantly improves

    Pros:
    - Very strong baseline (actual solutions, not estimates)
    - Automatically adapts as policy improves
    - No additional parameters to learn
    - Provides automatic curriculum (baseline gets harder over time)

    Cons:
    - Computationally expensive (requires additional rollouts)
    - Memory intensive (stores copy of model)
    - Can be slow to update in early training

    Reference: "Attention, Learn to Solve Routing Problems" (Kool et al., 2019)

    Args:
        model: Current policy model
        problem: Problem environment for generating datasets
        opts: Training options
        epoch: Current epoch number
    """

    def __init__(self, model, problem, opts, epoch=0):
        """
        Initialize the RolloutBaseline.

        Args:
            model: The policy model to use for rollouts.
            problem: The problem instance generator.
            opts: Configuration options.
            epoch (int): Current epoch (default: 0).
        """
        super(Baseline, self).__init__()
        self.problem = problem
        self.opts = opts
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        """
        Update the baseline model with a deep copy of the current model.

        Args:
            model: Current policy model.
            epoch (int): Current epoch.
            dataset: Optional dataset to use validation (default: generated).
        """
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts["val_size"]:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0]["loc"]).size(0) != self.opts["graph_size"]:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(
                area=self.opts["area"],
                waste_type=self.opts["waste_type"],
                size=self.opts["graph_size"],
                dist_matrix_path=self.opts["dm_filepath"],
                number_edges=self.opts["edge_threshold"],
                edge_strat=self.opts["edge_method"],
                focus_graph=self.opts["focus_graph"],
                focus_size=self.opts["eval_focus_size"],
                num_samples=self.opts["val_size"],
                distribution=self.opts["data_distribution"],
                vertex_strat=self.opts["vertex_method"],
                dist_strat=self.opts["distance_method"],
            )
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        """Wrap the dataset with rollout baseline values."""
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        """Unwrap the batch."""
        return batch["data"], batch["baseline"].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        """Evaluate the rollout baseline."""
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _, _ = self.model(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        print(
            "Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
                epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean
            )
        )
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = stats.ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts["bl_alpha"]:
                print("Update baseline")
                self._update_model(model, epoch)

    def state_dict(self):
        """Return state dict including model, dataset, and epoch."""
        return {"model": self.model, "dataset": self.dataset, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict["model"]).state_dict())
        self._update_model(load_model, state_dict["epoch"], state_dict["dataset"])


class BaselineDataset(torch.utils.data.Dataset):
    """
    dataset wrapping baseline values for training.

    Wraps the original dataset and corresponding baseline values to provide
    clean access during training batches.
    """

    def __init__(self, dataset=None, baseline=None):
        """
        Initialize the BaselineDataset.

        Args:
            dataset: Original training dataset.
            baseline: Tensor of baseline values corresponding to the dataset.
        """
        super(BaselineDataset, self).__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, item):
        """
        Get a data item and its baseline value.

        Args:
            item (int): Index.

        Returns:
            dict: {'data': ..., 'baseline': ...}
        """
        return {"data": self.dataset[item], "baseline": self.baseline[item]}

    def __len__(self):
        """
        Get the dataset length.

        Returns:
            int: Length.
        """
        return len(self.dataset)
