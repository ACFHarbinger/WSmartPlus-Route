"""Greedy evaluation implementation."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.utils.functions import move_to


class GreedyEval(EvalBase):
    """Greedy evaluation."""

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> dict:
        policy.eval()
        results = []
        start_time = time.time()

        for batch in tqdm(data_loader, disable=not self.progress, desc="Greedy Eval"):
            batch = move_to(batch, self.device)
            with torch.no_grad():
                out = policy(batch, decode_type="greedy", **kwargs)
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
