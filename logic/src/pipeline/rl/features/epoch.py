"""
Epoch-level utilities for RL training.
Includes expanded dataset handling and validation metric computation.
"""
from typing import Dict, Optional

import torch.nn as nn
from tensordict import TensorDict
from torch.utils.data import Dataset

from logic.src.data.datasets import GeneratorDataset


def prepare_epoch(
    model: nn.Module, env: any, baseline: any, dataset: Dataset, epoch: int, phase: str = "train"
) -> Dataset:
    """
    Prepare dataset for a new epoch.
    Handles baseline wrapping for RolloutBaseline.
    """
    if phase == "train" and hasattr(baseline, "wrap_dataset"):
        # Wrap dataset with baseline values (e.g. RolloutBaseline)
        return baseline.wrap_dataset(model, dataset, env)
    return dataset


def regenerate_dataset(
    env: any,
    size: int,
) -> Optional[Dataset]:
    """
    Regenerate training dataset using environment generator.
    """
    if hasattr(env, "generator"):
        return GeneratorDataset(env.generator, size)
    return None


def compute_validation_metrics(out: Dict, batch: TensorDict, env: any) -> Dict[str, float]:
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
    # Constructive environments usually have 'cost', 'len_cost', 'waste_cost' in State/Env
    # But usually we compute them post-hoc or via env.get_costs(batch, actions)

    # Check if env has cost function we can use
    if hasattr(env, "get_total_cost"):
        # cost: [batch]
        cost = env.get_total_cost(batch, out["actions"])
        metrics["val/total_cost"] = cost.mean().item()

    # 3. Efficiency (kg/km)
    # Need waste collected and distance traveled
    # Approximate if detailed breakdown not available

    # 4. Constraints Violations (Overflows)
    # If environment provides info on violations
    if hasattr(env, "get_num_overflows"):
        overflows = env.get_num_overflows(batch, out["actions"])
        metrics["val/overflows"] = overflows.float().mean().item()

    return metrics
