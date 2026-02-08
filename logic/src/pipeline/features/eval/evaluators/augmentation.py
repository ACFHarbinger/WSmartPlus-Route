"""Augmentation evaluation implementation."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.utils.functions import move_to


class AugmentationEval(EvalBase):
    """Evaluation with data augmentation (Test-Time Augmentation).
    Performs N augmentations and picks the best result for each sample.
    """

    def __init__(
        self,
        env: Any,
        num_augment: int = 8,
        augment_fn: str = "dihedral8",
        progress: bool = True,
        **kwargs,
    ):
        super().__init__(env, progress, **kwargs)
        from logic.src.data.transforms import StateAugmentation

        self.samples = num_augment
        self.augmentation = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> dict:
        policy.eval()
        results = []
        start_time = time.time()

        for batch in tqdm(data_loader, disable=not self.progress, desc=f"Augmentation Eval ({self.samples})"):
            batch = move_to(batch, self.device)
            with torch.no_grad():
                # Apply augmentation: returns [batch_size * num_augment, ...]
                aug_batch = self.augmentation(batch)
                out = policy(aug_batch, strategy="greedy", **kwargs)

                # Reshape and pick best from augmentations
                reward = out["reward"].view(self.samples, batch.batch_size[0], -1).max(0).values
                out["reward"] = reward
                results.append(out)

        total_time = time.time() - start_time
        avg_reward = torch.cat([r["reward"] for r in results]).mean().item()

        metrics = {
            "avg_reward": avg_reward,
            "total_time": total_time,
            "samples_per_second": len(data_loader.dataset) / total_time,
        }

        if return_results:
            metrics["results"] = results
        return metrics
