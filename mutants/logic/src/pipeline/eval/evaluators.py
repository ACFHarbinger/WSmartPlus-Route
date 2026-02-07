"""
Evaluator implementations.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
from logic.src.pipeline.eval.eval_base import EvalBase
from logic.src.utils.functions.function import move_to
from torch.utils.data import DataLoader
from tqdm import tqdm


class GreedyEval(EvalBase):
    """Greedy evaluation."""

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> Dict[str, Any]:
        policy.eval()
        policy.set_decode_type("greedy")

        start_time = time.time()
        rewards_list: List[torch.Tensor] = []
        sequences_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.progress, desc="Greedy Eval"):
                batch = move_to(batch, self.device)
                batch = self.env.reset(batch)
                out = policy(batch, self.env)
                rewards_list.append(out["reward"])
                if return_results:
                    sequences_list.append(out["actions"])

        duration = time.time() - start_time
        all_rewards = torch.cat(rewards_list)

        results = {
            "avg_reward": all_rewards.mean().item(),
            "std_reward": all_rewards.std().item(),
            "min_reward": all_rewards.min().item(),
            "max_reward": all_rewards.max().item(),
            "duration": duration,
        }
        if return_results:
            results["rewards"] = all_rewards
            results["sequences"] = torch.cat(sequences_list)
        return results


class SamplingEval(EvalBase):
    """Sampling evaluation with multiple samples."""

    def __init__(self, env: Any, samples: int = 1280, progress: bool = True, **kwargs):
        super().__init__(env, progress, **kwargs)
        self.samples = samples

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> Dict[str, Any]:
        policy.eval()
        policy.set_decode_type("sampling")

        start_time = time.time()
        rewards_list: List[torch.Tensor] = []
        sequences_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.progress, desc=f"Sampling {self.samples}"):
                batch = move_to(batch, self.device)
                batch = self.env.reset(batch)
                out = policy(batch, self.env, num_starts=self.samples)
                r = out["reward"]
                if r.dim() > 1:
                    # Best per instance
                    best_indices = r.max(dim=1)[1]
                    best_r = r.gather(1, best_indices[:, None]).squeeze(1)
                    if return_results:
                        s = out["actions"]
                        # actions is [batch, samples, seq_len]
                        s = s.gather(1, best_indices[:, None, None].expand(-1, -1, s.shape[-1])).squeeze(1)
                        sequences_list.append(s)
                    r = best_r
                else:
                    if return_results:
                        sequences_list.append(out["actions"])
                rewards_list.append(r)

        duration = time.time() - start_time
        all_rewards = torch.cat(rewards_list)
        results = {
            "avg_reward": all_rewards.mean().item(),
            "duration": duration,
        }
        if return_results:
            results["rewards"] = all_rewards
            results["sequences"] = torch.cat(sequences_list)
        return results


class AugmentationEval(EvalBase):
    """
    Evaluation with data augmentation (Test-Time Augmentation).
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

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> Dict[str, Any]:
        policy.eval()
        policy.set_decode_type("greedy")

        start_time = time.time()
        rewards_list: List[torch.Tensor] = []
        sequences_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.progress, desc=f"AugEval {self.samples}"):
                batch = move_to(batch, self.device)
                batch = self.env.reset(batch)
                batch_size = batch.batch_size[0]
                aug_batch = self.augmentation(batch)
                out = policy(aug_batch, self.env)
                r = out["reward"].view(batch_size, self.samples)
                best_indices = r.max(dim=1)[1]
                best_r = r.gather(1, best_indices[:, None]).squeeze(1)
                rewards_list.append(best_r)

                if return_results:
                    # actions is [batch*num_augment, seq_len]
                    s = out["actions"].view(batch_size, self.samples, -1)
                    best_s = s.gather(1, best_indices[:, None, None].expand(-1, -1, s.shape[-1])).squeeze(1)
                    sequences_list.append(best_s)

        duration = time.time() - start_time
        all_rewards = torch.cat(rewards_list)
        results = {
            "avg_reward": all_rewards.mean().item(),
            "duration": duration,
        }
        if return_results:
            results["rewards"] = all_rewards
            results["sequences"] = torch.cat(sequences_list)
        return results


class MultiStartEval(EvalBase):
    """Evaluation with multiple starts (POMO-style)."""

    def __init__(self, env: Any, num_starts: Optional[int] = None, progress: bool = True, **kwargs):
        super().__init__(env, progress, **kwargs)
        self.num_starts = num_starts

    def __call__(self, policy: Any, data_loader: DataLoader, **kwargs) -> Dict[str, float]:
        policy.eval()
        policy.set_decode_type("greedy")

        start_time = time.time()
        rewards_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.progress, desc="MultiStart Eval"):
                batch = self.env.reset(batch)
                num_starts = self.num_starts or batch["locs"].shape[-2]
                out = policy(batch, self.env, num_starts=num_starts)
                r = out["reward"]  # [batch, num_starts]
                r = r.max(dim=1)[0]
                rewards_list.append(r)

        duration = time.time() - start_time
        all_rewards = torch.cat(rewards_list)
        return {
            "avg_reward": all_rewards.mean().item(),
            "duration": duration,
        }


class MultiStartAugmentEval(EvalBase):
    """Combined Evaluation with multiple starts and data augmentation."""

    def __init__(
        self, env: Any, num_augment: int = 8, num_starts: Optional[int] = None, progress: bool = True, **kwargs
    ):
        super().__init__(env, progress, **kwargs)
        self.num_augment = num_augment
        self.num_starts = num_starts

    def __call__(self, policy: Any, data_loader: DataLoader, **kwargs) -> Dict[str, float]:
        from logic.src.data.transforms import StateAugmentation

        policy.eval()
        policy.set_decode_type("greedy")

        augment = StateAugmentation(num_augment=self.num_augment)
        start_time = time.time()
        rewards_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.progress, desc="MultiStart+Augment Eval"):
                batch = self.env.reset(batch)
                batch_aug = augment(batch)
                num_starts = self.num_starts or batch["locs"].shape[-2]
                out = policy(batch_aug, self.env, num_starts=num_starts)
                r = out["reward"]  # [batch * num_augment, num_starts]
                r = r.max(dim=1)[0]
                r = r.view(-1, self.num_augment).max(dim=1)[0]
                rewards_list.append(r)

        duration = time.time() - start_time
        all_rewards = torch.cat(rewards_list)
        return {
            "avg_reward": all_rewards.mean().item(),
            "duration": duration,
        }


# Aliases for ROADMAP parity
MultiStartGreedyEval = MultiStartEval
MultiStartGreedyAugmentEval = MultiStartAugmentEval
