"""
Evaluation Dispatcher.
"""

from typing import Any, Dict

from torch.utils.data import DataLoader

from logic.src.pipeline.features.eval.evaluators import (
    AugmentationEval,
    GreedyEval,
    MultiStartAugmentEval,
    MultiStartEval,
    SamplingEval,
)
from logic.src.utils.logging.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_automatic_batch_size(
    policy: Any,
    env: Any,
    data_loader: DataLoader,
    method: str = "greedy",
    initial_batch_size: int = 1024,
    max_tries: int = 10,
    **kwargs,
) -> int:
    """
    Automatically find the maximum batch size that fits in GPU memory.
    """
    import torch
    from torch.utils.data import Subset

    # Try a small subset first to find the batch size
    dataset = data_loader.dataset
    try:
        dataset_len = len(dataset)  # type: ignore
    except (TypeError, AttributeError):
        # Fallback for datasets without len (e.g. some IterableDatasets)
        dataset_len = initial_batch_size * 2

    subset_size = min(initial_batch_size * 2, dataset_len)
    subset_indices = list(range(subset_size))
    subset_dataset = Subset(data_loader.dataset, subset_indices)

    current_batch_size = initial_batch_size

    for i in range(max_tries):
        try:
            log.info(f"Testing batch size: {current_batch_size}")
            # Create a temporary loader with the current batch size
            temp_loader = DataLoader(subset_dataset, batch_size=current_batch_size, shuffle=False, num_workers=0)
            # Run one evaluation pass
            evaluate_policy(policy, env, temp_loader, method=method, **kwargs)
            log.info(f"Batch size {current_batch_size} works!")
            return current_batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if current_batch_size <= 1:
                    raise RuntimeError("Even a batch size of 1 causes OOM.")
                current_batch_size //= 2
                torch.cuda.empty_cache()
                log.warning(f"OOM detected. Reducing batch size to: {current_batch_size}")
            else:
                raise e

    raise RuntimeError("Could not find a valid batch size within max_tries.")


def evaluate_policy(
    policy: Any, env: Any, data_loader: DataLoader, method: str = "greedy", return_results: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a policy using the specified method.

    Args:
        policy: Policy to evaluate
        env: Environment
        data_loader: Dataset loader
        method: Decoding strategy ("greedy", "sampling", "augmentation", "multistart", "multistart_augment")
        **kwargs: Additional arguments for evaluator
            - samples (int): for sampling
            - num_augment (int): for augmentation
            - num_starts (int): for multistart
            - decoding (dict): Dictionary of decoding parameters, e.g., {"temperature": 1.0}

    Returns:
        Dict with metrics
    """
    device = kwargs.pop("device", "cpu")
    if method == "greedy":
        evaluator = GreedyEval(env, device=device, **kwargs)
    elif method == "sampling":
        evaluator = SamplingEval(env, device=device, **kwargs)
    elif method == "augmentation":
        evaluator = AugmentationEval(env, device=device, **kwargs)
    elif method == "multistart":
        evaluator = MultiStartEval(env, device=device, **kwargs)
    elif method == "multistart_augment":
        evaluator = MultiStartAugmentEval(env, device=device, **kwargs)
    else:
        # Fallback or error? For now default to greedy if unknown or raise
        raise ValueError(f"Unknown evaluation method: {method}")

    return evaluator(policy, data_loader, return_results=return_results)
