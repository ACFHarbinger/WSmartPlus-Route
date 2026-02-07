"""
Evaluator implementations.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.pipeline.features.eval_base import EvalBase


class GreedyEval(EvalBase):
    """Greedy evaluation."""

    def __call__(self, policy: Any, data_loader: DataLoader, **kwargs) -> Dict[str, float]:
        policy.eval()
        policy.set_decode_type("greedy")

        start_time = time.time()
        rewards = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.progress, desc="Greedy Eval"):
                batch = self.env.reset(batch)
                out = policy(batch, self.env)
                rewards.append(out["reward"])

        duration = time.time() - start_time
        rewards = torch.cat(rewards)
        return {
            "avg_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "min_reward": rewards.min().item(),
            "max_reward": rewards.max().item(),
            "duration": duration,
        }


class SamplingEval(EvalBase):
    """Sampling evaluation with multiple samples."""

    def __init__(self, env: Any, samples: int = 1280, progress: bool = True, **kwargs):
        super().__init__(env, progress, **kwargs)
        self.samples = samples

    def __call__(self, policy: Any, data_loader: DataLoader, **kwargs) -> Dict[str, float]:
        policy.eval()
        policy.set_decode_type("sampling")

        start_time = time.time()
        rewards = []

        # Note: Actual sampling logic might need efficient handling (multistart vs batch repetition)
        # This is a naive placeholder assuming policy handles sampling via decode_type
        # In WSmart/rl4co, typically we repeat the batch or use specialized decode method.
        # Assuming batch repetition for now if policy supports it via kwargs,
        # OR we just set decode_strategy in policy.

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.progress, desc=f"Sampling {self.samples}"):
                batch = self.env.reset(batch)

                # Check if policy supports passing num_starts/samples
                # If not, we might need to repeat batch manually.
                # Assuming standard WSmart/rl4co signature
                out = policy(batch, self.env, num_starts=self.samples)

                # out["reward"] should be [batch, samples]? Or max over samples?
                # Usually we want best per instance.
                r = out["reward"]
                if r.dim() > 1:
                    r = r.max(dim=1)[
                        0
                    ]  # Maximize reward (if reward is negative cost, works? No, usually minimize cost => max reward)
                    # WSmart: reward = -cost. So max reward is best.
                rewards.append(r)

        duration = time.time() - start_time
        rewards = torch.cat(rewards)
        return {
            "avg_reward": rewards.mean().item(),
            "duration": duration,
        }
