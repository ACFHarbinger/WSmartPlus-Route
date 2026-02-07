"""Multi-start evaluation implementation."""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.utils.functions import move_to
from torch.utils.data import DataLoader
from tqdm import tqdm


class MultiStartEval(EvalBase):
    """Evaluation with multiple starts (POMO-style)."""

    def __init__(self, env: Any, num_starts: Optional[int] = None, progress: bool = True, **kwargs):
        super().__init__(env, progress, **kwargs)
        self.num_starts = num_starts

    def __call__(self, policy: Any, data_loader: DataLoader, **kwargs) -> dict:
        policy.eval()
        results = []
        start_time = time.time()

        for batch in tqdm(data_loader, disable=not self.progress, desc="Multi-Start Eval"):
            batch = move_to(batch, self.device)
            with torch.no_grad():
                out = policy(batch, decode_type="greedy", num_starts=self.num_starts, **kwargs)
                results.append(out)

        total_time = time.time() - start_time
        avg_reward = torch.cat([r["reward"] for r in results]).mean().item()

        return {
            "avg_reward": avg_reward,
            "total_time": total_time,
            "samples_per_second": len(data_loader.dataset) / total_time,
        }
