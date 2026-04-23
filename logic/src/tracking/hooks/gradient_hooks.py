"""Gradient monitoring and diagnostic hooks for PyTorch models.

This module provides specialized backward hooks to inspect and influence the
backpropagation process. It includes utilities for detecting exploding or
vanishing gradients, automated clipping, NaN/Inf detection, and statistical
analysis of gradient flow across layers.

Attributes:
    add_gradient_monitoring_hooks: Utility to register diagnostic stats hooks.
    add_gradient_clipping_hook: Utility for per-layer gradient norm clipping.
    add_gradient_nan_detector_hook: Utility to trap numerical instability.

Example:
    >>> from logic.src.tracking.hooks.gradient_hooks import add_gradient_nan_detector_hook
    >>> handles = add_gradient_nan_detector_hook(model, raise_on_nan=True)
    >>> loss.backward()  # Raises ValueError if NaNs are found
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn


def add_gradient_monitoring_hooks(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    gradient_threshold: float = 10.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Registers backward hooks to monitor gradient statistics across layers.

    Detects vanishing/exploding gradients and provides layer-wise statistics.

    Args:
        model: PyTorch model to monitor.
        layer_names: Specific layer names to monitor. If None, monitors all.
            Defaults to None.
        gradient_threshold: Threshold for detecting exploding gradients.
            Defaults to 10.0.
        verbose: If True, prints warnings for abnormal gradients to stdout.
            Defaults to True.

    Returns:
        Dict[str, Any]: Mapping containing 'gradients' (stats) and 'handles'.
    """
    gradient_stats: List[Dict[str, Any]] = []
    hook_handles: List[Any] = []

    def gradient_hook(name: str) -> Callable[[torch.Tensor], None]:
        """Creates a backward hook that captures stats for a parameter.

        Args:
            name: Human-readable name for the parameter.

        Returns:
            Callable: The registered backward hook.
        """

        def hook(grad: torch.Tensor) -> None:
            """Internal hook that computes and records gradient metrics.

            Args:
                grad: The gradient tensor flowing backward.
            """
            if grad is None:
                return

            stats = {
                "name": name,
                "mean": grad.abs().mean().item(),
                "max": grad.abs().max().item(),
                "min": grad.abs().min().item(),
                "std": grad.std().item(),
                "norm": grad.norm().item(),
            }
            gradient_stats.append(stats)

            # Warn about potential issues
            if verbose:
                if stats["max"] > gradient_threshold:
                    print(f"⚠️  Exploding gradient in {name}: max={stats['max']:.2f}")
                if stats["mean"] < 1e-7:
                    print(f"⚠️  Vanishing gradient in {name}: mean={stats['mean']:.2e}")

        return hook

    # Register hooks on parameters
    for name, param in model.named_parameters():
        if param.requires_grad and (layer_names is None or name in layer_names):
            handle = param.register_hook(gradient_hook(name))
            hook_handles.append(handle)

    return {"gradients": gradient_stats, "handles": hook_handles}


def add_gradient_clipping_hook(
    model: nn.Module,
    max_norm: float = 1.0,
    layer_names: Optional[List[str]] = None,
) -> List[Any]:
    """Registers hooks to clip gradients per-layer during backward pass.

    Useful as an alternative to global gradient clipping when specific layers
    have gradient instability.

    Args:
        model: The target model.
        max_norm: Maximum gradient norm allowed per layer. Defaults to 1.0.
        layer_names: Specific layers to clip. If None, clips all trainable.
            Defaults to None.

    Returns:
        List[Any]: List of hook handles for later removal.
    """
    handles: List[Any] = []

    def clip_hook(norm_val: float) -> Callable[[torch.Tensor], torch.Tensor]:
        """Creates a hook that clips the gradient norm in-place.

        Args:
            norm_val: The clipping threshold.

        Returns:
            Callable: The registered clipping hook.
        """

        def hook(grad: torch.Tensor) -> torch.Tensor:
            """In-place gradient clipping hook.

            Args:
                grad: The incoming gradient.

            Returns:
                torch.Tensor: The clipped gradient tensor.
            """
            if grad is None:
                return grad
            return torch.nn.utils.clip_grad_norm_(grad, norm_val)

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad and (layer_names is None or name in layer_names):
            handle = param.register_hook(clip_hook(max_norm))
            handles.append(handle)

    return handles


def add_gradient_accumulation_hook(
    model: nn.Module,
    accumulation_steps: int = 4,
) -> Dict[str, Any]:
    """Registers hooks to track gradient accumulation statistics.

    Useful when using gradient accumulation to understand how gradients
    evolve and stabilize across multiple micro-batches.

    Args:
        model: The target model.
        accumulation_steps: Maximum history length of accumulation steps.
            Defaults to 4.

    Returns:
        Dict[str, Any]: Mapping with 'accumulated_norms' and 'handles'.
    """
    accumulated_norms: Dict[str, List[float]] = {name: [] for name, _ in model.named_parameters()}
    hook_handles: List[Any] = []

    def accumulation_hook(name: str) -> Callable[[torch.Tensor], None]:
        """Creates a hook to record the current gradient norm.

        Args:
            name: Parameter name.

        Returns:
            Callable: The registered accumulation hook.
        """

        def hook(grad: torch.Tensor) -> None:
            """Hook that records the norm and trims history to accumulation_steps.

            Args:
                grad: The current gradient.
            """
            if grad is not None:
                accumulated_norms[name].append(grad.norm().item())
                # Keep only last N accumulation steps
                if len(accumulated_norms[name]) > accumulation_steps:
                    accumulated_norms[name].pop(0)

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(accumulation_hook(name))
            hook_handles.append(handle)

    return {
        "accumulated_norms": accumulated_norms,
        "handles": hook_handles,
    }


def add_gradient_nan_detector_hook(
    model: nn.Module,
    raise_on_nan: bool = True,
) -> List[Any]:
    """Registers hooks to trap NaN/Inf values during backpropagation.

    Args:
        model: The target model.
        raise_on_nan: If True, raises ValueError immediately on detection.
            Defaults to True.

    Returns:
        List[Any]: List of hook handles.
    """
    handles: List[Any] = []

    def nan_detector_hook(name: str, raise_error: bool) -> Callable[[torch.Tensor], None]:
        """Creates a hook that scans the gradient for non-finite values.

        Args:
            name: Parameter name.
            raise_error: Whether to raise an exception.

        Returns:
            Callable: The registered detector hook.
        """

        def hook(grad: torch.Tensor) -> None:
            """Scans for NaN/Inf and raises or warns accordingly.

            Args:
                grad: The current gradient.

            Raises:
                ValueError: If NaN/Inf is detected and raise_error is True.
            """
            if grad is None:
                return

            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()

            if has_nan or has_inf:
                msg = f"NaN/Inf detected in gradients of {name}"
                if raise_error:
                    raise ValueError(msg)
                else:
                    print(f"⚠️  {msg}")

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(nan_detector_hook(name, raise_on_nan))
            handles.append(handle)

    return handles


def remove_all_hooks(hook_data: Dict[str, Any]) -> None:
    """Removes all registered hooks represented in the hook data mapping.

    Args:
        hook_data: Mapping returned by registration functions.
    """
    if "handles" in hook_data:
        for handle in hook_data["handles"]:
            handle.remove()
        hook_data["handles"].clear()


def print_gradient_statistics(gradient_stats: List[Dict[str, Any]], top_k: int = 10) -> None:
    """Prints a formatted summary of gradient magnitudes for analysis.

    Args:
        gradient_stats: Captured stats from monitoring hooks.
        top_k: Number of layers with largest norm to display. Defaults to 10.
    """
    if not gradient_stats:
        print("No gradient statistics available.")
        return

    # Sort by norm (descending)
    sorted_stats = sorted(gradient_stats, key=lambda x: x["norm"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"{'Layer Name':<40} {'Mean':>10} {'Max':>10} {'Norm':>10}")
    print(f"{'=' * 80}")

    for stat in sorted_stats[:top_k]:
        print(f"{stat['name']:<40} {stat['mean']:>10.6f} {stat['max']:>10.6f} {stat['norm']:>10.2f}")

    print(f"{'=' * 80}\n")
