"""PyTorch hooks and instrumentation utilities for WSmart-Route.

This package provides a comprehensive suite of hooks for monitoring, debugging,
and optimizing neural network training and inference. It enables developers to
capture intermediate activations, track gradient flow, monitor GPU memory
utilization, and analyze weight evolution with minimal boilerplate.

Attributes:
    add_attention_hooks: Utility to record attention head weights and masks.
    add_gradient_monitoring_hooks: Utility to diagnose gradient flow issues.
    add_activation_capture_hooks: Utility to record layer-wise activations.
    add_memory_profiling_hooks: Utility to profile GPU memory consumption.
    add_weight_change_monitor_hook: Utility to track parameter evolution.
    register_hooks_with_run: Utility to log hook summaries to a tracking Run.

Example:
    >>> from logic.src.tracking.hooks import add_gradient_monitoring_hooks
    >>> hook_data = add_gradient_monitoring_hooks(model)
    >>> loss.backward()
    >>> # Statistics are automatically recorded in hook_data
"""

from __future__ import annotations

from typing import Any, Dict

# Activation hooks
from logic.src.tracking.hooks.activation_hooks import (
    add_activation_capture_hooks,
    add_activation_sparsity_hook,
    add_activation_statistics_hook,
    add_dead_neuron_detector_hook,
    compute_activation_statistics,
    compute_sparsity_percentages,
    print_activation_summary,
)

# Attention hooks
from logic.src.tracking.hooks.attention_hooks import add_attention_hooks

# Gradient hooks
from logic.src.tracking.hooks.gradient_hooks import (
    add_gradient_accumulation_hook,
    add_gradient_clipping_hook,
    add_gradient_monitoring_hooks,
    add_gradient_nan_detector_hook,
    print_gradient_statistics,
)

# Memory hooks
from logic.src.tracking.hooks.memory_hooks import (
    add_memory_leak_detector_hook,
    add_memory_profiling_hooks,
    estimate_model_memory,
    optimize_batch_size,
    print_memory_summary,
)

# Weight hooks
from logic.src.tracking.hooks.weight_hooks import (
    add_weight_change_monitor_hook,
    add_weight_distribution_monitor,
    add_weight_norm_constraint_hook,
    add_weight_update_monitor_hook,
    analyze_weight_updates,
    compute_weight_changes,
    detect_weight_symmetry_breaking,
    print_weight_summary,
    restore_optimizer_step,
)

__all__ = [
    # Attention
    "add_attention_hooks",
    # Gradients
    "add_gradient_monitoring_hooks",
    "add_gradient_clipping_hook",
    "add_gradient_accumulation_hook",
    "add_gradient_nan_detector_hook",
    "print_gradient_statistics",
    # Activations
    "add_activation_capture_hooks",
    "add_activation_statistics_hook",
    "add_activation_sparsity_hook",
    "add_dead_neuron_detector_hook",
    "compute_activation_statistics",
    "compute_sparsity_percentages",
    "print_activation_summary",
    # Memory
    "add_memory_profiling_hooks",
    "add_memory_leak_detector_hook",
    "estimate_model_memory",
    "optimize_batch_size",
    "print_memory_summary",
    # Weights
    "add_weight_change_monitor_hook",
    "add_weight_distribution_monitor",
    "add_weight_update_monitor_hook",
    "add_weight_norm_constraint_hook",
    "compute_weight_changes",
    "detect_weight_symmetry_breaking",
    "restore_optimizer_step",
    "analyze_weight_updates",
    "print_weight_summary",
    # Tracking integration
    "register_hooks_with_run",
]


def register_hooks_with_run(hook_data: Dict[str, Any], run: Any, prefix: str = "hooks") -> None:  # noqa: C901
    """Logs hook statistics and diagnostic summaries to a tracking Run.

    This function parses the data dictionaries returned by hook registration
    functions and converts numeric summaries into flat parameters for the
    experiment manager. This allows visualization of gradient health and
    memory usage alongside hyper-parameters in the tracking UI.

    Args:
        hook_data: Dict mapping created by hook helpers. Must contain at least
            one of 'gradients', 'statistics', 'sparsity', 'memory_stats', or
            'dead_neurons'.
        run: An active tracking Run instance (e.g., MLflow, WandB, or WSTracker).
        prefix: Path prefix for generated parameter names. Defaults to 'hooks'.
    """
    if run is None:
        return

    params: Dict[str, Any] = {}

    # Gradient statistics
    gradients = hook_data.get("gradients", {})
    if isinstance(gradients, list):
        # Format from add_gradient_monitoring_hooks is a list of dicts
        for stat_entry in gradients:
            layer = stat_entry.get("name", "unknown")
            for stat_name, val in stat_entry.items():
                if stat_name != "name" and isinstance(val, (int, float)):
                    params[f"{prefix}/grad/{layer}/{stat_name}"] = round(float(val), 6)
    elif isinstance(gradients, dict):
        for layer, stats in gradients.items():
            for stat_name, val in stats.items():
                if isinstance(val, (int, float)):
                    params[f"{prefix}/grad/{layer}/{stat_name}"] = round(float(val), 6)

    # Activation statistics (already-computed final stats dict)
    statistics = hook_data.get("statistics", {})
    for layer, stats in statistics.items():
        for stat_name, val in stats.items():
            if isinstance(val, (int, float)):
                params[f"{prefix}/act/{layer}/{stat_name}"] = round(float(val), 6)

    # Sparsity
    sparsity = hook_data.get("sparsity", {})
    for layer, info in sparsity.items():
        if isinstance(info, dict):
            total = info.get("total", 0)
            zeros = info.get("zeros", 0)
            pct = zeros / total if total else 0.0
        else:
            pct = float(info)
        params[f"{prefix}/sparsity/{layer}"] = round(pct, 6)

    # Memory stats - summarise peak per layer
    memory_stats = hook_data.get("memory_stats", [])
    for stat in memory_stats:
        name = stat.get("name", "unknown")
        alloc = stat.get("allocated_mb", 0)
        params[f"{prefix}/mem/{name}/allocated_mb"] = round(float(alloc), 3)

    # Dead neurons
    dead_neurons = hook_data.get("dead_neurons", {})
    for layer, count in dead_neurons.items():
        params[f"{prefix}/dead/{layer}"] = int(count)

    if params:
        run.log_params(params)
