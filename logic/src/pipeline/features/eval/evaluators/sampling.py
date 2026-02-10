"""Sampling evaluation implementation."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.interfaces import ITraversable
from logic.src.utils.functions import do_batch_rep, move_to


class SamplingEval(EvalBase):
    """Sampling evaluation with multiple samples."""

    def __init__(self, env: Any, samples: int = 1280, progress: bool = True, **kwargs):
        """Initialize Class.

        Args:
            env (Any): Description of env.
            samples (int): Description of samples.
            progress (bool): Description of progress.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(env, progress, **kwargs)
        self.samples = samples

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> dict:
        """call  .

        Args:
            policy (Any): Description of policy.
            data_loader (DataLoader): Description of data_loader.
            return_results (bool): Description of return_results.
            kwargs (Any): Description of kwargs.

        Returns:
            Any: Description of return value.
        """
        policy.eval()
        results = []
        start_time = time.time()

        for batch in tqdm(data_loader, disable=not self.progress, desc=f"Sampling Eval ({self.samples} samples)"):
            batch = move_to(batch, self.device)

            # Expand batch for sampling
            batch_size = batch["loc"].size(0) if isinstance(batch, ITraversable) else batch.size(0)
            batch = do_batch_rep(batch, self.samples)

            with torch.no_grad():
                out = policy(batch, strategy="sampling", **kwargs)

                # Reshape outputs back to (batch_size, samples, ...)
                # Output from expanded batch is (batch_size * samples, ...)
                # grouped by samples of full batch: [b0, b1, ..., b0, b1, ...]
                # We want to group by instance: [[b0_s0, b0_s1...], [b1_s0, b1_s1...]]

                if "reward" in out:
                    # (Samples * Batch) -> (Samples, Batch) -> (Batch, Samples)
                    out["reward"] = out["reward"].view(self.samples, batch_size).transpose(0, 1)

                if "actions" in out:
                    # (Samples * Batch, Len) -> (Samples, Batch, Len) -> (Batch, Samples, Len)
                    out["actions"] = out["actions"].view(self.samples, batch_size, -1).transpose(0, 1)

                results.append(out)

        total_time = time.time() - start_time
        avg_reward = torch.cat([r["reward"] for r in results]).mean().item()

        metrics = {
            "avg_reward": avg_reward,
            "duration": total_time,
            "samples_per_second": len(data_loader.dataset) / total_time,
        }

        if return_results:
            # Calculate best from samples
            # results: list of dicts. r["reward"]: (B, S), r["actions"]: (B, S, L)
            all_rewards = torch.cat([r["reward"] for r in results], dim=0)  # (Total, S)

            # Pad sequences to max length
            max_len = max([r["actions"].size(-1) for r in results])
            padded_actions = []
            for r in results:
                act = r["actions"]
                pad_len = max_len - act.size(-1)
                if pad_len > 0:
                    act = torch.nn.functional.pad(act, (0, pad_len), value=0)
                padded_actions.append(act)
            all_actions = torch.cat(padded_actions, dim=0)  # (Total, S, L)

            # Max over samples dim
            max_val, max_idx = all_rewards.max(dim=1)
            best_rewards = max_val

            # Gather best sequences
            # all_actions: (B, S, L). max_idx: (B).
            # We want (B, L).
            max_idx_expanded = max_idx.view(-1, 1, 1).expand(-1, 1, all_actions.size(2))
            best_sequences = all_actions.gather(1, max_idx_expanded).squeeze(1)

            metrics["rewards"] = best_rewards.cpu()
            metrics["sequences"] = best_sequences.cpu()
            metrics["results"] = results
        return metrics
