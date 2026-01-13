"""
Base Trainer Module for REINFORCE-based Training Loops.

This module provides an abstract base class implementing the Template Method design pattern
for REINFORCE-style policy gradient training. It defines the common training loop structure
while allowing subclasses to customize specific steps like dataset initialization and
day/epoch training logic.

The Template Method pattern ensures consistency across different training variants
(Standard, Time-based, HRL, Meta-learning) while enabling customization where needed.
"""
import torch
import numpy as np

from abc import ABC, abstractmethod
from logic.src.utils.log_utils import log_epoch
from logic.src.utils.visualize_utils import visualize_epoch
from logic.src.pipeline.reinforcement_learning.core.epoch import complete_train_pass


class BaseReinforceTrainer(ABC):
    """
    Abstract base trainer implementing the Template Method pattern for REINFORCE training loops.

    This class provides the skeleton of a REINFORCE training algorithm, defining the overall
    training flow while delegating specific steps to subclasses. It manages the training loop,
    logging, checkpointing, and coordination with meta-learners for dynamic weight adjustment.

    The training loop follows this structure:
    1. Setup meta-learner (if applicable)
    2. Initialize training dataset
    3. For each day/epoch:
        a. Update context (meta-learning hook)
        b. Train for one day/epoch (implemented by subclasses)
        c. Post-day processing (logging, visualization, checkpointing)
        d. Process feedback (meta-learning hook)

    Attributes:
        model: The neural network model to train
        optimizer: PyTorch optimizer for model parameters
        baseline: Baseline object for variance reduction in policy gradients
        lr_scheduler: Learning rate scheduler (optional)
        scaler: GradScaler for mixed precision training (optional)
        val_dataset: Validation dataset
        problem: Problem environment (VRPP, WCVRP, etc.)
        tb_logger: TensorBoard logger
        cost_weights: Dictionary of weights for multi-objective cost function
        opts: Training options/hyperparameters dictionary
        step: Global training step counter
        day: Current training day/epoch
        weight_optimizer: Meta-learner for dynamic weight adjustment (optional)
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the BaseReinforceTrainer.

        Args:
            model: Neural network model.
            optimizer: Optimizer.
            baseline: Baseline for updates.
            lr_scheduler: Learning rate scheduler.
            scaler: GradScaler for mixed precision.
            val_dataset: Validation dataset.
            problem: Problem instance.
            tb_logger: TensorBoard logger.
            cost_weights: Cost weights dict.
            opts: Options dictionary.
        """
        self.model = model
        self.optimizer = optimizer
        self.baseline = baseline
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.val_dataset = val_dataset
        self.problem = problem
        self.tb_logger = tb_logger
        self.cost_weights = cost_weights
        self.opts = opts
        
        self.step = 0
        self.day = opts['epoch_start']
        self.weight_optimizer = None # Meta-learner

    def setup_meta_learner(self):
        """
        Initialize the meta-learner if needed.

        This hook is called once at the beginning of training to set up any meta-learning
        components (e.g., weight optimizers, contextual bandits, hypernetworks).
        Subclasses that use meta-learning should override this method.

        The default implementation does nothing.
        """
        pass

    def update_context(self):
        """
        Update context/weights before the day/epoch starts.

        This hook is called at the beginning of each training day/epoch, before train_day().
        It allows the trainer to:
        - Update cost function weights based on meta-learning
        - Adjust hyperparameters dynamically
        - Update the training dataset (for time-based training)
        - Prepare any context-specific state

        The default implementation does nothing.
        """
        pass

    def process_feedback(self):
        """
        Provide feedback to the meta-learner after training step/day.

        This hook is called at the end of each training day/epoch, after post_day_processing().
        It allows the trainer to:
        - Send performance metrics to meta-learners
        - Update meta-learning models based on training results
        - Adjust strategies for the next iteration

        The default implementation does nothing.
        """
        pass

    def train(self):
        """
        Main training loop (Template Method).

        This method orchestrates the entire training process using the Template Method pattern.
        It defines the skeleton of the algorithm while allowing subclasses to customize
        specific steps through hook methods.

        Flow:
        1. setup_meta_learner() - Initialize meta-learning components
        2. Synchronize initial weights if meta-learner exists
        3. initialize_training_dataset() - Prepare training data
        4. For each day/epoch until max_days:
            a. update_context() - Pre-day hook
            b. train_day() - Execute one day of training (implemented by subclasses)
            c. post_day_processing() - Logging, visualization, checkpointing
            d. process_feedback() - Meta-learning feedback
            e. Check stopping criteria

        This method should not be overridden by subclasses.
        """
        self.setup_meta_learner()
        
        # Initial weight synchronization if meta-learner exists
        if self.weight_optimizer:
            if hasattr(self.weight_optimizer, 'propose_weights'):
                pass
            elif hasattr(self.weight_optimizer, 'get_current_weights'):
                weights = self.weight_optimizer.get_current_weights()
                self.cost_weights.update(weights)

        # Initialize dataset 
        self.initialize_training_dataset()

        # Training loop
        max_days = self.opts['epoch_start'] + self.opts['n_epochs']
        while self.day < max_days:
            # Hook: Pre-day updates
            self.update_context()

            # Train for a single day/epoch
            self.train_day()
            
            # Hook: Post-day processing
            self.post_day_processing()
            self.process_feedback()

            self.day += 1
            if self._should_stop():
                break

    def initialize_training_dataset(self):
        """
        Initialize or load the training dataset.

        This hook is called once before the training loop starts. Subclasses should override
        this method to:
        - Load or generate the training dataset
        - Set up time-dependent data structures (for time-based training)
        - Configure data-specific parameters

        The default implementation does nothing (assumes dataset is set elsewhere or not needed).
        """
        pass

    @abstractmethod
    def train_day(self):
        """
        Execute training for a single day/epoch (iterate over dataloader).

        This is the core training method that must be implemented by all subclasses.
        It should:
        1. Iterate over batches in the training dataset
        2. For each batch:
            - Forward pass through the model
            - Compute loss (REINFORCE, PPO, or variant)
            - Backward pass and optimizer step
        3. Accumulate metrics for logging

        Expected side effects:
        - Updates self.daily_loss with loss metrics
        - Updates self.step counter
        - May update self.log_pi and self.log_costs for time-based training

        This method is called once per day/epoch by the train() template method.
        """
        pass

    def post_day_processing(self):
        """
        Common post-day processing: Logging, Visualization, Checkpointing.

        This method is called automatically after each train_day() execution. It handles:
        1. Logging metrics to TensorBoard and console (via log_epoch)
        2. Visualization of routes/solutions (if enabled via visualize_step option)
        3. Model validation on val_dataset
        4. Checkpointing model state
        5. Learning rate scheduling
        6. Baseline updates

        Subclasses can override this method to add custom post-processing, but should
        typically call super().post_day_processing() to maintain standard behavior.

        Expects:
        - self.daily_loss to be set by train_day()
        - self.day_duration to be set by train_day() (optional)
        """
        # Note: self.daily_loss must be set by train_day
        if hasattr(self, 'daily_loss'):
            log_epoch(('day', self.day), list(self.daily_loss.keys()), self.daily_loss, self.opts)
        
        # Visualization Hook
        if self.opts.get('visualize_step', 0) > 0 and (self.day + 1) % self.opts['visualize_step'] == 0:
            visualize_epoch(self.model, self.problem, self.opts, self.day, tb_logger=self.tb_logger)
            
        _ = complete_train_pass(
            self.model, self.optimizer, self.baseline, self.lr_scheduler, self.val_dataset,
            self.day, self.step, getattr(self, 'day_duration', 0), self.tb_logger, self.cost_weights, self.opts, 
            manager=self.weight_optimizer
        )

        # --- Curriculum Update Logic ---
        if self.opts.get('imitation_weight', 0.0) > 0:
            # 1. Decay
            if (self.day + 1) % self.opts['imitation_decay_step'] == 0:
                self.opts['imitation_weight'] *= self.opts['imitation_decay']
                print(f"[Curriculum] Decay Step: Imitation Weight updated to {self.opts['imitation_weight']:.6f}")
                
            # 2. Reannealing
            # Safe extraction helper using torch
            def get_mean_metric(key):
                """Get mean metric value from daily loss"""
                if key not in self.daily_loss: return 0.0
                vals = self.daily_loss[key]
                if not vals: return 0.0
                # vals is a list of tensors or floats
                if isinstance(vals[0], torch.Tensor):
                    return torch.cat(vals).float().mean().item()
                return np.mean(vals)
            
            avg_expert_cost = get_mean_metric('expert_cost')
            
            # Implementation detail: Use a running counter stored in self
            if not hasattr(self, 'reannealing_counter'):
                self.reannealing_counter = 0
                self.initial_im_weight = self.opts.get('imitation_weight', 0.0)

            # Reconstruct model cost robustly since 'total' mixes cost and loss
            avg_length = get_mean_metric('length') * self.cost_weights['length']
            avg_overflows = get_mean_metric('overflows') * self.cost_weights['overflows']
            avg_waste = get_mean_metric('waste') * self.cost_weights['waste']
            avg_model_cost = avg_length + avg_overflows + avg_waste
            print(f"[Curriculum] Reannealing: Model Cost {avg_model_cost:.4f} vs Expert Cost {avg_expert_cost:.4f}")
            
            threshold = self.opts.get('reannealing_threshold', 0.05)
            
            # print(f"[Curriculum] Reannealing: Model Cost {avg_model_cost:.4f} vs Expert Cost {avg_expert_cost:.4f}")

            if avg_model_cost > avg_expert_cost * (1 + threshold):
                self.reannealing_counter += 1
            else:
                self.reannealing_counter = 0
                
            if self.reannealing_counter >= self.opts.get('reannealing_patience', 5):
                print(f"[Curriculum] Reannealing: Resetting imitation weight to {self.initial_im_weight}")
                self.opts['imitation_weight'] = self.initial_im_weight
                self.reannealing_counter = 0
            # -------------------------------

    def _should_stop(self):
        """
        Check if training should stop early.

        This internal method is called at the end of each training day/epoch to determine
        if training should terminate before reaching max_days. Subclasses can override this
        to implement custom stopping criteria (e.g., convergence detection, performance thresholds).

        Returns:
            bool: True if training should stop, False to continue
        """
        return False

    def train_batch(self, batch, batch_id, opt_step=True):
        """
        Train on a single batch.

        This method contains the core logic for processing a single batch. Most trainers
        inherit this implementation from StandardTrainer in reinforce.py, but it's defined
        here as a placeholder.

        Args:
            batch: Dictionary containing batch data (inputs, targets, etc.)
            batch_id: Index of the current batch in the dataloader
            opt_step: If True, perform optimizer step; if False, only compute gradients

        Returns:
            tuple: (pi, c_dict, l_dict, cost, state_tensors)
                - pi: Action sequences/tours generated by the model
                - c_dict: Dictionary of cost components (overflows, kg, km, etc.)
                - l_dict: Dictionary of loss components (total, reinforce_loss, baseline_loss, etc.)
                - cost: Scalar or tensor representing total cost
                - state_tensors: Dictionary of intermediate tensors (for off-policy algorithms)

        Raises:
            NotImplementedError: Subclasses must implement or inherit this method
        """
        raise NotImplementedError
