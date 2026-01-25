"""
Epoch-level utilities for RL training.
Includes expanded dataset handling and validation metric computation.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
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
        return baseline.wrap_dataset(dataset, model, env)
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

    Args:
        out: Output dictionary from policy (contains 'reward', 'actions', etc.)
        batch: Input TensorDict batch
        env: Environment instance

    Returns:
        metrics: Dictionary of scalar metrics
    """
    metrics = {}

    # 1. Main Reward
    if "reward" in out:
        metrics["val/reward"] = out["reward"].mean().item()

    # 2. Costs Breakdown (if available)
    # Check if env has detailed cost function
    if hasattr(env, "get_costs"):
        # Expecting a dict of tensors: {'waste': ..., 'dist': ..., 'overflow': ...}
        # We assume out["actions"] is present
        if "actions" in out:
            costs = env.get_costs(batch, out["actions"])
            for key, val in costs.items():
                if isinstance(val, torch.Tensor):
                    metrics[f"val/{key}"] = val.float().mean().item()

    # 3. Efficiency (kg/km)
    # If not already computed in get_costs
    if "val/efficiency" not in metrics:
        if "val/waste" in metrics and "val/dist" in metrics:
            # Avoid division by zero
            avg_waste = metrics["val/waste"]
            avg_dist = metrics["val/dist"]
            if avg_dist > 1e-6:
                metrics["val/efficiency"] = avg_waste / avg_dist

    # 4. Constraints Violations (Overflows)
    # If environment provides info on violations
    if hasattr(env, "get_num_overflows") and "actions" in out:
        overflows = env.get_num_overflows(batch, out["actions"])
        metrics["val/overflows"] = overflows.float().mean().item()

    return metrics
