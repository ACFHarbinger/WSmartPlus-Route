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
                out = policy(batch, strategy="greedy", **kwargs)
                results.append(out)

        total_time = time.time() - start_time
        avg_reward = torch.cat([r["reward"] for r in results]).mean().item()

        metrics = {
            "avg_reward": avg_reward,
            "duration": total_time,
            "samples_per_second": len(data_loader.dataset) / total_time,
        }

        if return_results:
            # Assuming output has 'reward' and 'actions' (or 'sequences'?)
            # AttentionModel output usually has 'reward', 'log_likelihood', 'actions'
            metrics["rewards"] = torch.cat([r["reward"] for r in results]).cpu()

            # Pad sequences to max length
            max_len = max([r["actions"].size(-1) for r in results])
            padded_actions = []
            for r in results:
                act = r["actions"]
                pad_len = max_len - act.size(-1)
                if pad_len > 0:
                    act = torch.nn.functional.pad(act, (0, pad_len), value=0)
                padded_actions.append(act)
            metrics["sequences"] = torch.cat(padded_actions).cpu()
            metrics["results"] = results
        return metrics
