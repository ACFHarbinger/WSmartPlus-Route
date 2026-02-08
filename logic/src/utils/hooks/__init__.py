"""
PyTorch Hooks Utilities for WSmart-Route.

This module provides comprehensive hooks for monitoring, debugging, and optimizing
neural network training. Hooks can capture intermediate outputs, track gradients,
monitor memory usage, and analyze weight updates.

Usage Categories:
    - Attention Hooks: Capture attention weights and masks
    - Gradient Hooks: Monitor gradient flow and detect vanishing/exploding gradients
    - Activation Hooks: Track activations, dead neurons, and sparsity
    - Memory Hooks: Profile GPU memory usage and detect leaks
    - Weight Hooks: Monitor weight updates and distribution changes

Example:
    >>> from logic.src.utils.hooks import (
    ...     add_gradient_monitoring_hooks,
    ...     add_activation_capture_hooks,
    ...     add_memory_profiling_hooks,
    ... )
    >>>
    >>> # Monitor gradients during training
    >>> grad_hooks = add_gradient_monitoring_hooks(model)
    >>> loss.backward()
    >>> print_gradient_statistics(grad_hooks['gradients'])
    >>>
    >>> # Capture activations
    >>> act_hooks = add_activation_capture_hooks(model, layer_types=(nn.Linear,))
    >>> output = model(input)
    >>> for name, activation in act_hooks['activations'].items():
    ...     print(f"{name}: {activation.shape}")
"""

from __future__ import annotations

# Activation hooks
from logic.src.utils.hooks.activation_hooks import (
    add_activation_capture_hooks,
    add_activation_sparsity_hook,
    add_activation_statistics_hook,
    add_dead_neuron_detector_hook,
    compute_activation_statistics,
    compute_sparsity_percentages,
    print_activation_summary,
)

# Attention hooks
from logic.src.utils.hooks.attention import add_attention_hooks

# Gradient hooks
from logic.src.utils.hooks.gradient_hooks import (
    add_gradient_accumulation_hook,
    add_gradient_clipping_hook,
    add_gradient_monitoring_hooks,
    add_gradient_nan_detector_hook,
    print_gradient_statistics,
)

# Memory hooks
from logic.src.utils.hooks.memory_hooks import (
    add_memory_leak_detector_hook,
    add_memory_profiling_hooks,
    estimate_model_memory,
    optimize_batch_size,
    print_memory_summary,
)

# Weight hooks
from logic.src.utils.hooks.weight_hooks import (
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
]
