"""Runtime attention ring-buffer for decoder introspection (§A.2 Option A).

Captures multi-head attention matrices from encoder forward hooks and retains
the most recent snapshots for Studio ML introspection and eval-time emission.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


class AttentionRingBuffer:
    """Fixed-capacity ring-buffer for encoder attention snapshots.

    Each snapshot records layer index, head index, decode step, node count, and
    a row-normalised attention matrix suitable for JSON serialisation.
    """

    def __init__(self, max_history: int = 64, max_matrix_dim: int = 128) -> None:
        """Initialise buffer capacity and matrix size guardrails."""
        self._snapshots: List[Dict[str, Any]] = []
        self._max = max_history
        self._max_dim = max_matrix_dim
        self._decode_step = 0

    @property
    def decode_step(self) -> int:
        """Current autoregressive decode step index."""
        return self._decode_step

    def bump_decode_step(self) -> None:
        """Advance decode step after each routing decision."""
        self._decode_step += 1

    def record(
        self,
        layer_idx: int,
        tensor: torch.Tensor,
        head_idx: int = 0,
    ) -> None:
        """Append one attention matrix snapshot when shape is supported."""
        from logic.src.tracking.logging.visualization.heatmaps import extract_attention_matrix

        matrix = extract_attention_matrix(tensor, head_idx=head_idx)
        if matrix is None:
            return
        n_nodes = int(matrix.shape[0])
        if n_nodes < 2 or n_nodes > self._max_dim:
            return

        entry = {
            "layer": layer_idx,
            "head": head_idx,
            "decode_step": self._decode_step,
            "n_nodes": n_nodes,
            "matrix": np.round(matrix, 4).tolist(),
        }
        self._snapshots.append(entry)
        if len(self._snapshots) > self._max:
            self._snapshots.pop(0)

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of buffered snapshots."""
        return [dict(s) for s in self._snapshots]

    def as_layer_matrices(self, head_idx: int = 0) -> List[Tuple[int, np.ndarray]]:
        """Convert snapshots to ``(layer_idx, matrix)`` pairs for heatmap logging."""
        result: List[Tuple[int, np.ndarray]] = []
        seen_layers: set[int] = set()
        for snap in self._snapshots:
            if snap.get("head", 0) != head_idx:
                continue
            layer = int(snap["layer"])
            if layer in seen_layers:
                continue
            seen_layers.add(layer)
            matrix = np.asarray(snap["matrix"], dtype=np.float32)
            result.append((layer, matrix))
        return result

    def reset(self, *, reset_decode_step: bool = True) -> None:
        """Discard buffered snapshots."""
        self._snapshots.clear()
        if reset_decode_step:
            self._decode_step = 0

    def __len__(self) -> int:
        return len(self._snapshots)


def install_attention_ring_buffer(
    encoder: nn.Module,
    buffer: AttentionRingBuffer,
    head_idx: int = 0,
) -> List[Any]:
    """Register persistent forward hooks that append to ``buffer``."""
    handles: List[Any] = []

    def make_hook(layer_idx: int):
        def hook(module: nn.Module, _input: Tuple[torch.Tensor, ...], _output: Any) -> None:
            last_attn = getattr(module, "last_attn", None)
            if last_attn is None:
                return
            try:
                weights = last_attn[0]
            except (IndexError, TypeError):
                return
            buffer.record(layer_idx, weights, head_idx=head_idx)

        return hook

    for layer_idx, layer in enumerate(getattr(encoder, "layers", [])):
        if not hasattr(layer, "att"):
            continue
        attention_module = layer.att.module
        handles.append(attention_module.register_forward_hook(make_hook(layer_idx)))

    return handles


def ensure_attention_buffer(
    model: Any,
    head_idx: int = 0,
) -> Optional[AttentionRingBuffer]:
    """Attach a ring-buffer and persistent hooks to ``model`` when possible."""
    policy = model
    if hasattr(model, "policy"):
        policy = model.policy

    buffer = getattr(policy, "attention_buffer", None)
    if buffer is None:
        buffer = AttentionRingBuffer()
        policy.attention_buffer = buffer  # type: ignore[attr-defined]

    encoder = None
    for attr in ("embedder", "encoder"):
        candidate = getattr(policy, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            encoder = candidate
            break

    if encoder is None:
        return buffer

    handles = getattr(policy, "_attention_hook_handles", None)
    if not handles:
        policy._attention_hook_handles = install_attention_ring_buffer(  # type: ignore[attr-defined]
            encoder,
            buffer,
            head_idx=head_idx,
        )

    return buffer
