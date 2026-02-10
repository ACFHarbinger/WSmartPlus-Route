"""
Epoch-level utilities for RL training.
Includes expanded dataset handling and validation metric computation.
"""

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.data import Dataset


def prepare_epoch(
    model: nn.Module,
    env: Any,
    baseline: Any,
    dataset: Dataset,
    epoch: int,
    phase: str = "train",
) -> Dataset:
    """
    Prepare dataset for a new epoch.
    Handles baseline wrapping for RolloutBaseline.
    """
    if phase == "train" and hasattr(baseline, "wrap_dataset"):
        # Unwrap dataset first to avoid nested BaselineDataset
        if hasattr(baseline, "unwrap_dataset"):
            dataset = baseline.unwrap_dataset(dataset)
        # Wrap dataset with baseline values (e.g. RolloutBaseline)
        return baseline.wrap_dataset(dataset, policy=model, env=env)
    return dataset


def regenerate_dataset(
    env: Any,
    size: int,
) -> Optional[Dataset]:
    """
    Regenerate training dataset using environment generator.
    """
    if hasattr(env, "generator"):
        # Pre-generate for efficiency
        from logic.src.data.datasets import TensorDictDataset

        gen = env.generator
        if hasattr(gen, "to"):
            gen = gen.to("cpu")
        data = gen(batch_size=size)
        return TensorDictDataset(data)
    return None


def compute_validation_metrics(out: Dict, batch: TensorDict, env: Any) -> Dict[str, float]:
    """
    Compute rich validation metrics beyond simple reward.
    """
    metrics: Dict[str, float] = {}
    _add_reward_metric(metrics, out)
    _add_costs_metrics(metrics, out, batch, env)
    _add_efficiency_metric(metrics)
    _add_overflow_metrics(metrics, out, batch, env)
    return metrics


def _add_reward_metric(metrics: Dict[str, float], out: Dict) -> None:
    """Add mean reward to metrics."""
    if "reward" in out:
        metrics["val/reward"] = out["reward"].mean().item()


def _add_costs_metrics(metrics: Dict[str, float], out: Dict, batch: TensorDict, env: Any) -> None:
    """Add detailed cost breakdown from environment if available."""
    if hasattr(env, "get_costs") and "actions" in out:
        costs = env.get_costs(batch, out["actions"])
        for key, val in costs.items():
            if isinstance(val, torch.Tensor):
                metrics[f"val/{key}"] = val.float().mean().item()


def _add_efficiency_metric(metrics: Dict[str, float]) -> None:
    """Compute efficiency ratio (kg/km) from waste and distance."""
    if "val/efficiency" not in metrics and "val/waste" in metrics and "val/dist" in metrics:
        avg_waste = metrics["val/waste"]
        avg_dist = metrics["val/dist"]
        if avg_dist > 1e-6:
            metrics["val/efficiency"] = avg_waste / avg_dist


def _add_overflow_metrics(metrics: Dict[str, float], out: Dict, batch: TensorDict, env: Any) -> None:
    """Add constraint violation counts (overflows) if available."""
    if hasattr(env, "get_num_overflows") and "actions" in out:
        overflows = env.get_num_overflows(batch, out["actions"])
        metrics["val/overflows"] = overflows.float().mean().item()
