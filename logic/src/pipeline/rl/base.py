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
        batch_size: int = 256,
        num_workers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env", "policy"])

        self.env = env
        self.policy = policy
        self.baseline_type = baseline

        # Data params
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
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
        from logic.src.pipeline.rl.baselines import get_baseline

        self.baseline = get_baseline(self.baseline_type, self.policy)

    @abstractmethod
    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
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
        td = self.env.reset(batch)

        # Run policy
        out = self.policy(
            td,
            self.env,
            decode_type="sampling" if phase == "train" else "greedy",
        )

        # Compute loss for training
        if phase == "train":
            out["loss"] = self.calculate_loss(td, out, batch_idx)

        # Log metrics
        reward_mean = out["reward"].mean()
        self.log(f"{phase}/reward", reward_mean, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/cost", -reward_mean, sync_dist=True)

        if "log_likelihood" in out:
            self.log(f"{phase}/log_likelihood", out["log_likelihood"].mean(), sync_dist=True)

        return out

    def training_step(self, batch: TensorDict, batch_idx: int) -> torch.Tensor:
        out = self.shared_step(batch, batch_idx, phase="train")
        return out["loss"]

    def validation_step(self, batch: TensorDict, batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: TensorDict, batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="test")

    def on_train_epoch_end(self):
        """Update baseline at epoch end."""
        if hasattr(self.baseline, "epoch_callback"):
            self.baseline.epoch_callback(self.policy, self.current_epoch)

    def configure_optimizers(self):
        """Configure optimizer and optional scheduler."""
        # Get optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.policy.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.lr_scheduler_name is None:
            return optimizer

        # Get scheduler
        if self.lr_scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.lr_scheduler_kwargs)
        elif self.lr_scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage: Optional[str] = None):
        """Initialize datasets."""
        self.train_dataset = GeneratorDataset(
            self.env.generator,
            self.train_data_size,
        )
        self.val_dataset = GeneratorDataset(
            self.env.generator,
            self.val_data_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=self.num_workers,
        )
