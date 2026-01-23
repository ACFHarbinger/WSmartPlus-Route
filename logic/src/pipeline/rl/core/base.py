"""
PyTorch Lightning base module for RL training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from logic.src.data.datasets import GeneratorDataset, tensordict_collate_fn
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.base import ConstructivePolicy


class RL4COLitModule(pl.LightningModule, ABC):
    """
    Base PyTorch Lightning module for RL training.

    This module handles:
    - Training/validation/test loops
    - Optimizer configuration
    - Data loading
    - Metric logging
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: ConstructivePolicy,
        baseline: Optional[str] = "rollout",
        optimizer: str = "adam",
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        val_dataset_path: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        **kwargs,
    ):
        """
        Initialize the RL4COLitModule.

        Args:
            env: The RL environment for the problem.
            policy: The neural network policy.
            baseline: Type of baseline for variance reduction ('rollout', 'exponential', 'critic', etc.).
            optimizer: Optimizer name ('adam' or 'adamw').
            optimizer_kwargs: Keyword arguments for the optimizer.
            lr_scheduler: Learning rate scheduler name ('cosine', 'step', or None).
            lr_scheduler_kwargs: Keyword arguments for the scheduler.
            train_data_size: Number of training samples to generate per epoch.
            val_data_size: Number of validation samples.
            val_dataset_path: Optional path to a pre-saved validation dataset.
            batch_size: Batch size for training and validation.
            num_workers: Number of workers for data loading.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["env", "policy"])

        self.env = env
        self.policy = policy
        self.baseline_type = baseline

        # Data params
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Optimizer params
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Initialize baseline
        self._init_baseline()

    def _init_baseline(self):
        """Initialize baseline for advantage estimation."""
        from logic.src.pipeline.rl.core.baselines import WarmupBaseline, get_baseline

        baseline = get_baseline(self.baseline_type, self.policy, **self.hparams)

        # Handle warmup
        warmup_epochs = self.hparams.get("bl_warmup_epochs", 0)
        if warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, warmup_epochs)

        self.baseline = baseline

    @abstractmethod
    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional[RL4COEnvBase] = None,
    ) -> torch.Tensor:
        """
        Compute RL loss.

        Args:
            td: TensorDict with environment state.
            out: Policy output dictionary.
            batch_idx: Current batch index.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        Common step for train/val/test.

        Args:
            batch: TensorDict batch.
            batch_idx: Batch index.
            phase: One of "train", "val", "test".

        Returns:
            Output dictionary with loss, reward, etc.
        """
        # Unwrap batch if it's from a baseline dataset
        batch, baseline_val = self.baseline.unwrap_batch(batch)
        batch = batch.to(self.device)
        if baseline_val is not None:
            baseline_val = baseline_val.to(self.device)
        self._current_baseline_val = baseline_val

        td = self.env.reset(batch)

        # Run policy
        out = self.policy(
            td,
            self.env,
            decode_type="sampling" if phase == "train" else "greedy",
        )

        # Compute loss for training
        if phase == "train":
            out["loss"] = self.calculate_loss(td, out, batch_idx, env=self.env)

        # Merge granular metrics from td if available
        for key in ["collection", "cost"]:
            if key in list(td.keys()):
                out[key] = td[key]

        # Log metrics
        reward_mean = out["reward"].mean()
        self.log(
            f"{phase}/reward",
            reward_mean,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.batch_size[0],
        )

        if "collection" in out:
            self.log(
                f"{phase}/collection",
                out["collection"].mean(),
                sync_dist=True,
                batch_size=batch.batch_size[0],
            )
        if "cost" in out:
            self.log(
                f"{phase}/cost",
                out["cost"].mean(),
                sync_dist=True,
                batch_size=batch.batch_size[0],
            )
        else:
            self.log(f"{phase}/cost_total", -reward_mean, sync_dist=True)

        # Store for meta-learning or logging access
        self.last_out = out

        return out

    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            batch: Input batch, potentially wrapped by baseline.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        # 1. Unwrap batch if it was wrapped by baseline (e.g. RolloutBaseline)
        if hasattr(self.baseline, "unwrap_batch"):
            td, baseline_val = self.baseline.unwrap_batch(batch)
        else:
            td, baseline_val = batch, None

        # 2. Run shared step
        out = self.shared_step(td, batch_idx, phase="train")

        # 3. Calculate loss with baseline_val if available
        # Pass baseline_val to calculate_loss if needed
        # We'll update calculate_loss signature later or use a private attribute
        self._current_baseline_val = baseline_val

        return out["loss"]

    def validation_step(self, batch: TensorDict, batch_idx: int) -> dict:
        """
        Execute a single validation step.

        Args:
            batch: TensorDict batch for validation.
            batch_idx: Index of the current batch.

        Returns:
            dict: Output dictionary with metrics.
        """
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: TensorDict, batch_idx: int) -> dict:
        """
        Execute a single test step.

        Args:
            batch: TensorDict batch for testing.
            batch_idx: Index of the current batch.

        Returns:
            dict: Output dictionary with metrics.
        """
        return self.shared_step(batch, batch_idx, phase="test")

    def on_train_epoch_start(self) -> None:
        """Prepare dataset for the new epoch (e.g. wrap with baseline)."""
        from logic.src.pipeline.rl.features.epoch import prepare_epoch

        self.train_dataset = prepare_epoch(
            self.policy,
            self.env,
            self.baseline,
            self.train_dataset,
            self.current_epoch,
            phase="train",
        )

    def on_train_epoch_end(self):
        """Update baseline and regenerate dataset."""
        from logic.src.pipeline.rl.features.epoch import regenerate_dataset

        if hasattr(self.baseline, "epoch_callback"):
            # For RolloutBaseline, we pass val_dataset for the T-test
            self.baseline.epoch_callback(
                self.policy,
                self.current_epoch,
                val_dataset=self.val_dataset,
                env=self.env,
            )

        # Regenerate training dataset for next epoch if configured
        if self.hparams.get("regenerate_per_epoch", False):
            if self.trainer.max_epochs is not None and self.current_epoch < self.trainer.max_epochs - 1:
                # Check if environment supports generation
                if hasattr(self.env, "generator"):
                    new_dataset = regenerate_dataset(self.env, self.train_data_size)
                    if new_dataset is not None:
                        self.train_dataset = new_dataset

                    # Log generation
                    if self.local_rank == 0:
                        print(f"Regenerated training dataset for epoch {self.current_epoch + 1}")

    def configure_optimizers(self):
        """Configure optimizer and optional scheduler."""
        # Get optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.policy.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.lr_scheduler_name is None:
            return optimizer

        # Get scheduler
        name = self.lr_scheduler_name.lower()
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.lr_scheduler_kwargs)
        elif name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_scheduler_kwargs)
        elif name == "lambda":
            gamma = self.lr_scheduler_kwargs.get("gamma", 1.0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: gamma**epoch)
        elif name == "exp":
            gamma = self.lr_scheduler_kwargs.get("gamma", 0.99)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage: str) -> None:
        """
        Set up datasets for training and validation.

        Args:
            stage: The stage ('fit', 'validate', 'test', or 'predict').
        """
        if stage == "fit":
            from torch.utils.data import Dataset

            # If num_workers > 0, we should generate on CPU to avoid CUDA fork issues
            # and transfer to device in shared_step.
            generator = self.env.generator
            if self.num_workers > 0 and generator.device.type != "cpu":
                if hasattr(generator, "to"):
                    generator = generator.to("cpu")
                else:
                    # Fallback or warning if .to() is not available (though we just added it)
                    print("Warning: Generator on GPU with num_workers > 0. Forces CPU generator.")
                    # Ideally we wouldn't reach here if we assume generators.py is updated.

            self.train_dataset: Dataset = GeneratorDataset(
                generator,
                self.train_data_size,
            )
            if self.val_dataset_path is not None:
                # Load validation dataset from file (legacy parity)
                from logic.src.data.datasets import TensorDictDataset

                self.val_dataset: Dataset = TensorDictDataset.load(self.val_dataset_path)
            else:
                self.val_dataset = GeneratorDataset(
                    generator,  # Use same CPU generator for validation if generated
                    self.val_data_size,
                )

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=self.num_workers,
        )
