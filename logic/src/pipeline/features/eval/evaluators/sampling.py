"""Sampling evaluation implementation."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.utils.functions import move_to


class SamplingEval(EvalBase):
    """Sampling evaluation with multiple samples."""

    def __init__(self, env: Any, samples: int = 1280, progress: bool = True, **kwargs):
        super().__init__(env, progress, **kwargs)
        self.samples = samples

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> dict:
        policy.eval()
        results = []
        start_time = time.time()

        for batch in tqdm(data_loader, disable=not self.progress, desc=f"Sampling Eval ({self.samples} samples)"):
            batch = move_to(batch, self.device)
            with torch.no_grad():
                # We need to repeat the batch for sampling
                # This could be memory-intensive for large batches/sample sizes
                out = policy(batch, decode_type="sampling", num_samples=self.samples, **kwargs)
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
