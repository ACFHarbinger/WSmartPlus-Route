from abc import ABC, abstractmethod

from logic.src.utils.log_utils import log_epoch
from logic.src.pipeline.reinforcement_learning.core.epoch import complete_train_pass
from logic.src.utils.visualize_utils import visualize_epoch

class BaseReinforceTrainer(ABC):
    """
    Abstract base trainer implementing the Template Method pattern for REINFORCE training loops.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
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
        """
        pass

    def update_context(self):
        """
        Update context/weights before the day/epoch starts.
        """
        pass

    def process_feedback(self):
        """
        Provide feedback to the meta-learner after training step/day.
        """
        pass

    def train(self):
        """
        Main training loop (Template Method).
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
        pass

    @abstractmethod
    def train_day(self):
        """
        Execute training for a single day (iterate over dataloader).
        Must be implemented by subclasses.
        """
        pass

    def post_day_processing(self):
        """
        Common post-day processing: Logging, Visualization, Checkpointing.
        Subclasses can override, but usually calling super() or this implementation is sufficient.
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

    def _should_stop(self):
        return False

    def train_batch(self, batch, batch_id, opt_step=True):
        # Logic extracted from train_batch_reinforce in reinforce.py
        # This seems to be shared logic used by most trainers.
        # Moving it here for reuse if possible, or keep in specific trainers?
        # Trainers typically inherit from BaseReinforceTrainer, so having it here or in a Mixin is good.
        # But 'train_day' calls 'train_batch'.
        # Let's put abstract method here or implement default.
        raise NotImplementedError
