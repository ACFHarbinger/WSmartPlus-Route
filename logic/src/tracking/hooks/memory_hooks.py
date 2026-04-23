"""Memory profiling and monitoring hooks for PyTorch models.

This module provides utilities to record and analyze GPU memory consumption
during model execution. It includes hooks for per-layer memory tracking,
automated leak detection, and offline estimation of model footprints. These
tools are essential for optimizing batch sizes and identifying memory-hungry
attention mechanisms in large routing networks.

Attributes:
    add_memory_profiling_hooks: Utility to record per-layer memory snapshots.
    add_memory_leak_detector_hook: Utility to warn on memory growth over time.
    estimate_model_memory: Theoretical calculation of memory requirements.
    optimize_batch_size: Empirical search for maximizing GPU utilization.

Example:
    >>> from logic.src.tracking.hooks.memory_hooks import add_memory_profiling_hooks
    >>> hook_data = add_memory_profiling_hooks(model)
    >>> _ = model(input_tensor)
    >>> print_memory_summary(hook_data['memory_stats'])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


def add_memory_profiling_hooks(
    model: nn.Module,
    device: Optional[torch.device] = None,
    track_allocations: bool = True,
) -> Dict[str, Any]:
    """Registers forward hooks to track memory usage per layer.

    Args:
        model: PyTorch model to profile.
        device: Device to monitor (auto-detected if None). Defaults to None.
        track_allocations: If True, tracks active CUDA memory in addition to
            reserved pool. Defaults to True.

    Returns:
        Dict[str, Any]: Mapping containing 'memory_stats' and 'handles'.
    """
    if device is None:
        device = next(model.parameters()).device

    memory_stats: List[Dict[str, Any]] = []
    hook_handles: List[Any] = []

    def memory_hook(name: str, target_device: torch.device) -> Callable:
        """Creates a hook that captures memory usage after layer execution.

        Args:
            name: Layer identifier.
            target_device: Device being monitored.

        Returns:
            Callable: The registered memory hook.
        """

        def hook(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Any) -> None:
            """Internal forward hook that captures CUDA memory statistics.

            Args:
                module: The layer generating the output.
                input_tensors: Layer inputs.
                output: Layer outputs.
            """
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
                allocated = torch.cuda.memory_allocated(target_device) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(target_device) / 1024**2  # MB
                max_allocated = torch.cuda.max_memory_allocated(target_device) / 1024**2  # MB

                stat = {
                    "name": name,
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "max_allocated_mb": max_allocated,
                }
                memory_stats.append(stat)

        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        if name:  # Skip root module
            handle = module.register_forward_hook(memory_hook(name, device))
            hook_handles.append(handle)

    return {"memory_stats": memory_stats, "handles": hook_handles}


def add_memory_leak_detector_hook(
    model: nn.Module,
    threshold_mb: float = 100.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Registers hooks to detect cumulative memory growth across forward passes.

    Args:
        model: The model to monitor.
        threshold_mb: Memory growth threshold (in MB) to trigger a warning.
            Defaults to 100.0.
        device: Device to monitor (auto-detected if None). Defaults to None.

    Returns:
        Dict[str, Any]: Mapping containing 'memory_history' and 'handles'.
    """
    if device is None:
        device = next(model.parameters()).device

    memory_history: List[Dict[str, Any]] = []
    hook_handles: List[Any] = []
    state: Dict[str, Optional[float]] = {"initial_memory": None}

    def leak_detector_hook(name: str, target_device: torch.device, threshold: float) -> Callable:
        """Creates a hook that identifies unauthorized memory accumulation.

        Args:
            name: Layer identifier.
            target_device: Monitoring device.
            threshold: Warning threshold in MB.

        Returns:
            Callable: The registered detector hook.
        """

        def hook(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Any) -> None:
            """Forward hook that detects memory leaks between passes.

            Args:
                module: The layer being executed.
                input_tensors: Layer inputs.
                output: Layer outputs.
            """
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
                current_memory = torch.cuda.memory_allocated(target_device) / 1024**2

                if state["initial_memory"] is None:
                    state["initial_memory"] = current_memory

                initial = state["initial_memory"]
                if initial is not None:
                    growth = current_memory - initial
                    memory_history.append({"name": name, "memory_mb": current_memory, "growth_mb": growth})

                    if growth > threshold:
                        print(f"⚠️  Potential memory leak in {name}: {growth:.2f} MB growth")

        return hook

    # Register hook on root module to track overall growth
    handle = model.register_forward_hook(leak_detector_hook("model", device, threshold_mb))
    hook_handles.append(handle)

    return {"memory_history": memory_history, "handles": hook_handles}


def estimate_model_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Estimates the theoretical memory footprint of a model and training state.

    Args:
        model: PyTorch model.
        input_shape: Shape of a single input sample (excluding batch dimension).
        batch_size: Evaluation batch size. Defaults to 1.
        dtype: Numerical data type. Defaults to torch.float32.

    Returns:
        Dict[str, float]: Memory estimates in MB for parameters, activations,
            gradients, and optimizer states.
    """
    # Calculate parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    # Estimate activation memory (rough approximation)
    activation_memory = param_memory * batch_size * 0.5  # Rough estimate

    # Input memory
    bytes_per_element = torch.finfo(dtype).bits // 8
    input_memory = batch_size * torch.prod(torch.tensor(input_shape)).item() * bytes_per_element / 1024**2

    # Gradient memory (same as parameters)
    gradient_memory = param_memory

    # Optimizer state (Adam: 2x parameters)
    optimizer_memory = param_memory * 2

    return {
        "param_mb": float(param_memory),
        "activation_mb": float(activation_memory),
        "input_mb": float(input_memory),
        "gradient_mb": float(gradient_memory),
        "optimizer_mb": float(optimizer_memory),
        "total_mb": float(param_memory + activation_memory + input_memory + gradient_memory + optimizer_memory),
    }


def print_memory_summary(memory_stats: List[Dict[str, Any]], top_k: int = 10) -> None:
    """Renders a formatted table of memory consumption per module.

    Args:
        memory_stats: Snapshot list from add_memory_profiling_hooks.
        top_k: Number of memory-consuming layers to display. Defaults to 10.
    """
    if not memory_stats:
        print("No memory statistics available.")
        return

    # Sort by allocated memory (descending)
    sorted_stats = sorted(memory_stats, key=lambda x: x["allocated_mb"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"{'Layer Name':<40} {'Allocated':>15} {'Reserved':>15}")
    print(f"{'=' * 80}")

    for stat in sorted_stats[:top_k]:
        print(f"{stat['name']:<40} {stat['allocated_mb']:>12.2f} MB {stat['reserved_mb']:>12.2f} MB")

    total_allocated = sorted_stats[0]["allocated_mb"] if sorted_stats else 0.0
    total_reserved = sorted_stats[0]["reserved_mb"] if sorted_stats else 0.0

    print(f"{'=' * 80}")
    print(f"{'TOTAL':<40} {total_allocated:>12.2f} MB {total_reserved:>12.2f} MB")
    print(f"{'=' * 80}\n")


def optimize_batch_size(
    model: nn.Module,
    input_generator: Callable[[int], Any],
    initial_batch_size: int = 32,
    device: Optional[torch.device] = None,
    safety_margin: float = 0.9,
) -> int:
    """Heuristically searches for the maximum batch size the GPU can support.

    Runs binary search on batch sizes, executing a forward pass at each step
    until an Out-of-Memory (OOM) error is encountered.

    Args:
        model: The model to test.
        input_generator: Function returning inputs for a given batch size.
        initial_batch_size: Starting point for the search. Defaults to 32.
        device: Target device (must be CUDA). Defaults to None.
        safety_margin: Scaling factor applied to the maximum found size.
            Defaults to 0.9.

    Returns:
        int: Recommended 'safe' batch size.
    """
    if device is None:
        device = torch.device("cuda")
    model.to(device)
    model.eval()

    max_working_batch_size = initial_batch_size

    print(f"Testing batch sizes starting from {initial_batch_size}...")

    # Binary search for maximum batch size
    low, high = 1, initial_batch_size * 16
    while low <= high:
        mid = (low + high) // 2

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            # Test forward pass
            with torch.no_grad():
                inputs = input_generator(mid)
                _ = model(inputs)

            torch.cuda.synchronize(device)
            max_working_batch_size = mid
            print(f"  Batch size {mid}: ✓ (Peak: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB)")

            # Try larger
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch size {mid}: ✗ (OOM)")
                # Try smaller
                high = mid - 1
            else:
                raise e

    optimal_batch_size = int(max_working_batch_size * safety_margin)
    print(f"\nOptimal batch size (with {safety_margin * 100:.0f}% safety margin): {optimal_batch_size}")

    return optimal_batch_size


def remove_all_hooks(hook_data: Dict[str, Any]) -> None:
    """Detaches all PyTorch handles stored in the hook data dictionary.

    Args:
        hook_data: Mapping returned by any memory registration function.
    """
    if "handles" in hook_data:
        for handle in hook_data["handles"]:
            handle.remove()
        hook_data["handles"].clear()
