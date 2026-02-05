"""
PyTorch Lightning base module for RL training.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, cast

import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from logic.src.data.datasets import tensordict_collate_fn
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.base import ConstructivePolicy
from logic.src.policies.selection import VectorizedSelector
from logic.src.utils.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


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
        persistent_workers: bool = True,
        pin_memory: bool = False,
        must_go_selector: Optional[VectorizedSelector] = None,
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
            train_dataset_path: Optional path to a pre-saved training dataset.
            batch_size: Batch size for training and validation.
            num_workers: Number of workers for data loading.
            must_go_selector: Optional vectorized selector for must-go bin selection.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["env", "policy", "must_go_selector"])

        self.env = env
        self.policy = policy
        self.baseline_type = baseline
        self.train_dataset: Optional[Any] = None
        self.must_go_selector = must_go_selector

        # Data params
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.val_dataset_path = val_dataset_path
        self.train_dataset_path = kwargs.get("train_dataset_path")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        # Optimizer params
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Initialize baseline
        self._init_baseline()

    def save_weights(self, path: str):
        """
        Save model weights and hyperparameters.

        Args:
            path: Path to save the weights to.
        """
        import json
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save weights and hparams to .pt file
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hparams": self.hparams,
            },
            path,
        )

        # Save hparams to sidecar args.json
        args_path = os.path.join(os.path.dirname(path), "args.json")

        # Convert hparams to a serializable dict
        # Filter out non-serializable objects (env, policy) if they were not ignored
        hparams_dict: Dict[str, Any] = {}
        # ... existing logic to populate hparams_dict would go here if not already present
        # For now, we ensure it's annotated to avoid Mypy errors
        try:
            with open(args_path, "w") as f:
                json.dump(hparams_dict, f, indent=4)
        except (OSError, TypeError, ValueError) as e:
            logger.warning(f"Could not save sidecar args.json: {e}")

        # Save full Hydra config if available
        if hasattr(self, "cfg") and self.cfg is not None:
            config_path = os.path.join(os.path.dirname(path), "config.yaml")
            try:
                from omegaconf import OmegaConf

                OmegaConf.save(config=self.cfg, f=config_path)
            except (OSError, ImportError, TypeError, ValueError) as e:
                logger.warning(f"Could not save config.yaml: {e}")

    def _init_baseline(self):
        """Initialize baseline for advantage estimation."""
        from logic.src.pipeline.rl.common.baselines import WarmupBaseline, get_baseline

        if self.baseline_type is None:
            # Default to rollout if None, or raise error?
            # get_baseline likely expects string.
            self.baseline_type = "rollout"
        baseline = get_baseline(self.baseline_type, self.policy, **self.hparams)

        # Handle warmup
        warmup_epochs = self.hparams.get("bl_warmup_epochs", 0)
        if warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, warmup_epochs)

        self.baseline = baseline

    def _apply_must_go_selection(self, td: TensorDict) -> TensorDict:
        """
        Apply must-go selection to determine which bins must be collected.

        Args:
            td: TensorDict with problem instance data.

        Returns:
            TensorDict with 'must_go' mask added.
        """
        if self.must_go_selector is None:
            return td

        # Get fill levels from the TensorDict
        # WCVRP uses 'demand', VRPP uses 'prize' or 'demand'
        fill_levels = None
        for key in ["demand", "prize", "fill_level"]:
            if key in td.keys():
                fill_levels = td[key]
                break

        if fill_levels is None:
            logger.warning("No fill levels found in TensorDict for must-go selection")
            return td

        # Ensure fill_levels is 2D (batch_size, num_nodes)
        if fill_levels.dim() == 1:
            fill_levels = fill_levels.unsqueeze(0)

        # Get additional data for advanced selectors
        selector_kwargs = {}
        if "accumulation_rate" in td.keys():
            selector_kwargs["accumulation_rates"] = td["accumulation_rate"]
        if "std_deviation" in td.keys():
            selector_kwargs["std_deviations"] = td["std_deviation"]
        if "current_day" in td.keys():
            selector_kwargs["current_day"] = td["current_day"]

        # Apply selector to get must-go mask
        must_go_mask = self.must_go_selector.select(fill_levels, **selector_kwargs)

        # Store in TensorDict
        td["must_go"] = must_go_mask

        return td

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
        batch: Union[TensorDict, Dict[str, Any]],
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

        # Move to device (crucial when pin_memory=False)
        if hasattr(batch, "to"):
            batch = cast(Any, batch).to(self.device)
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}

        if baseline_val is not None:
            baseline_val = cast(Any, baseline_val).to(self.device)
        self._current_baseline_val = baseline_val

        # env.reset expects data on the environment's device.
        from logic.src.utils.functions.rl import ensure_tensordict

        td = ensure_tensordict(batch, self.device)

        # Apply must-go selector if configured
        if self.must_go_selector is not None:
            td = self._apply_must_go_selection(td)

        td = self.env.reset(td)

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
        batch_size = out["reward"].shape[0]

        self.log(
            f"{phase}/reward",
            reward_mean,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if "collection" in out:
            self.log(
                f"{phase}/collection",
                out["collection"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
        if "cost" in out:
            self.log(
                f"{phase}/cost",
                out["cost"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
        else:
            self.log(f"{phase}/cost_total", -reward_mean, sync_dist=True)

        # Store for meta-learning or logging access
        self.last_out = out

        return out

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
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

    def validation_step(self, batch: Any, batch_idx: int) -> dict:  # type: ignore[override]
        """
        Execute a single validation step.

        Args:
            batch: TensorDict batch for validation.
            batch_idx: Index of the current batch.

        Returns:
            dict: Output dictionary with metrics.
        """
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int) -> dict:  # type: ignore[override]
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
        from logic.src.pipeline.rl.common.epoch import prepare_epoch

        assert self.train_dataset is not None
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
        from logic.src.pipeline.rl.common.epoch import regenerate_dataset

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
                        logger.info(f"Regenerated training dataset for epoch {self.current_epoch + 1}")

    def configure_optimizers(self) -> Any:  # type: ignore[override]
        """Configure optimizer and optional scheduler."""
        optimizer: Any = None
        # Get optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.policy.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        opt: Any = optimizer

        if self.lr_scheduler_name is None:
            return optimizer

        scheduler: Any = None
        # Get scheduler
        name = self.lr_scheduler_name.lower()
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, **self.lr_scheduler_kwargs)
        elif name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, **self.lr_scheduler_kwargs)
        elif name == "lambda":
            gamma = self.lr_scheduler_kwargs.get("gamma", 1.0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: gamma**epoch)
        elif name == "exp":
            gamma = self.lr_scheduler_kwargs.get("gamma", 0.99)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        elif name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")

        return {"optimizer": opt, "lr_scheduler": scheduler}

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
            from logic.src.data.datasets import TensorDictDataset

            # Pre-generate dataset on CPU for efficiency and VRAM saving
            # This avoids the overhead of generating 1 instance at a time in __getitem__
            gen = self.env.generator
            assert gen is not None
            if hasattr(gen, "to"):
                gen = gen.to("cpu")

            if self.local_rank == 0:
                logger.info(f"Pre-generating training dataset ({self.train_data_size} instances) on CPU...")

            assert gen is not None
            if self.train_dataset_path is not None and os.path.exists(self.train_dataset_path):
                if self.local_rank == 0:
                    logger.info(f"Loading training dataset from {self.train_dataset_path}")
                from logic.src.data.datasets import TensorDictDataset

                self.train_dataset = TensorDictDataset.load(self.train_dataset_path)
            else:
                data = gen(batch_size=self.train_data_size)
                self.train_dataset = TensorDictDataset(data)

            assert self.train_dataset is not None
            if self.val_dataset_path is not None:
                # Load validation dataset from file (legacy parity)
                from logic.src.data.datasets import TensorDictDataset

                self.val_dataset: Dataset = TensorDictDataset.load(self.val_dataset_path)
            else:
                if self.local_rank == 0:
                    logger.info(f"Pre-generating validation dataset ({self.val_data_size} instances) on CPU...")
                val_data = cast(Any, gen)(batch_size=self.val_data_size)
                self.val_dataset = TensorDictDataset(val_data)
                assert self.val_dataset is not None
        else:
            pass

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(
            cast(Any, self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            collate_fn=tensordict_collate_fn,
            pin_memory=self.pin_memory if self.num_workers > 0 and self.train_dataset is not None else False,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0 and self.train_dataset is not None
            else False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        assert self.val_dataset is not None
        return DataLoader(
            cast(Any, self.val_dataset),
            batch_size=self.batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory if self.num_workers > 0 and self.val_dataset is not None else False,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0 and self.val_dataset is not None
            else False,
        )
