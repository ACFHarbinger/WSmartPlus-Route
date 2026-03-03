"""
ONNX Export Utility for WSmart-Route neural encoders and policies.

Safely traces and exports custom Encoder architectures (GAT, GCN, GGAC, MLP)
and full attention-based policies to ``.onnx`` format for graph inspection
in Netron (https://netron.app).

Key design decisions:
- Uses ``torch.onnx.export`` with dynamic axes to support variable batch/graph sizes.
- Wraps encoders in a minimal shim to normalise tuple outputs (some encoders return
  ``(embeddings, mask)`` tuples).
- Validates the exported model via ``onnx.checker`` before returning.
- Optional ``onnxsim`` simplification pass to reduce graph complexity for Netron.

CLI usage:
    python -m logic.src.utils.model.export_onnx \\
        --checkpoint assets/model_weights/best.pt \\
        --output_dir assets/onnx/ \\
        --num_loc 50 \\
        --embed_dim 128 \\
        --simplify

Programmatic usage:
    from logic.src.utils.model.export_onnx import export_encoder_to_onnx
    path = export_encoder_to_onnx(model.encoder, export_dir="assets/onnx", n_nodes=50)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_encoder_to_onnx(
    encoder: nn.Module,
    export_dir: str = "assets/onnx",
    filename: Optional[str] = None,
    n_nodes: int = 50,
    embed_dim: int = 128,
    batch_size: int = 1,
    opset_version: int = 17,
    simplify: bool = False,
    verbose: bool = False,
    generator: Optional[torch.Generator] = None,
) -> str:
    """
    Trace and export a single encoder ``nn.Module`` to ONNX format.

    Creates a synthetic dummy input of shape ``(batch_size, n_nodes, embed_dim)``
    and exports with dynamic batch and node-count axes so the Netron graph is
    readable at any problem scale.

    Args:
        encoder: The encoder ``nn.Module`` to export (e.g., GATEncoder, GCNEncoder).
        export_dir: Directory where the ``.onnx`` file will be written.
        filename: Output filename stem (without extension). Defaults to
            ``encoder.__class__.__name__.lower()``.
        n_nodes: Number of graph nodes in the dummy input. Default 50.
        embed_dim: Node feature dimension for the dummy input. Default 128.
        batch_size: Dummy batch size (1 is sufficient for Netron). Default 1.
        opset_version: ONNX opset version. Minimum 13 required for scatter ops
            used in GNN layers. Default 17.
        simplify: If True, run ``onnxsim.simplify`` for a cleaner Netron graph.
            Requires ``pip install onnxsim``.
        verbose: Enable verbose ONNX export logging.
        generator: Optional PyTorch generator for reproducible random inputs.

    Returns:
        Absolute path to the written ``.onnx`` file.

    Raises:
        RuntimeError: If tracing or ONNX validation fails.
    """
    os.makedirs(export_dir, exist_ok=True)
    model_name = filename or encoder.__class__.__name__.lower()
    out_path = str(Path(export_dir) / f"{model_name}.onnx")

    encoder.eval()
    device = _infer_device(encoder)
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(42)

    dummy_input = torch.randn(batch_size, n_nodes, embed_dim, device=device, generator=generator)

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "node_embeddings": {0: "batch_size", 1: "n_nodes"},
        "output": {0: "batch_size", 1: "n_nodes"},
    }

    wrapped = _SingleInputWrapper(encoder)

    log.info("Tracing %s → %s", encoder.__class__.__name__, out_path)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped,
                dummy_input,
                out_path,
                input_names=["node_embeddings"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                verbose=verbose,
                do_constant_folding=True,
            )
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed for {encoder.__class__.__name__}: {exc}") from exc

    _validate_onnx(out_path)

    if simplify:
        _try_simplify(out_path)

    log.info("Exported: %s", out_path)
    return os.path.abspath(out_path)


def export_policy_components(
    policy: nn.Module,
    export_dir: str = "assets/onnx",
    n_nodes: int = 50,
    embed_dim: int = 128,
    batch_size: int = 1,
    opset_version: int = 17,
    simplify: bool = False,
    component_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Export the named sub-modules of a policy to separate ONNX files.

    Iterates over the policy's direct children, filtering by ``component_names``
    (default: ``["encoder", "decoder"]``), and exports each independently.

    Args:
        policy: The parent policy ``nn.Module`` (e.g., AttentionModel policy).
        export_dir: Output directory for all ``.onnx`` files.
        n_nodes: Graph size for dummy inputs.
        embed_dim: Node embedding dimension.
        batch_size: Dummy batch size.
        opset_version: ONNX opset version.
        simplify: Apply onnxsim simplification.
        component_names: Which child module names to export. Defaults to
            ``["encoder", "decoder"]``.

    Returns:
        Dict mapping component name → absolute path of the exported file.
    """
    target_names = set(component_names or ["encoder", "decoder"])
    results: Dict[str, str] = {}

    for name, module in policy.named_children():
        if name not in target_names:
            continue
        policy_cls = policy.__class__.__name__.lower()
        try:
            path = export_encoder_to_onnx(
                encoder=module,
                export_dir=export_dir,
                filename=f"{policy_cls}_{name}",
                n_nodes=n_nodes,
                embed_dim=embed_dim,
                batch_size=batch_size,
                opset_version=opset_version,
                simplify=simplify,
            )
            results[name] = path
        except RuntimeError as exc:
            log.warning("Skipped %s.%s: %s", policy_cls, name, exc)

    if not results:
        log.warning(
            "No exportable components found in %s. Looked for: %s. Available children: %s",
            policy.__class__.__name__,
            sorted(target_names),
            [n for n, _ in policy.named_children()],
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _SingleInputWrapper(nn.Module):
    """
    Minimal shim that accepts a single tensor input and returns a single tensor.

    Required because some encoders return ``(embeddings, mask)`` tuples or
    accept additional keyword arguments that confuse ``torch.onnx.export``.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.encoder(x)
        # Unwrap tuple/list outputs — return first tensor element
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _infer_device(module: nn.Module) -> torch.device:
    """Return the device of the first parameter, defaulting to CPU."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _validate_onnx(path: str) -> None:
    """Run onnx.checker on the exported model. Logs a warning if onnx is not installed."""
    try:
        import onnx  # type: ignore[import]

        model = onnx.load(path)
        onnx.checker.check_model(model)
        log.info("ONNX validation passed for %s", path)
    except ImportError:
        log.warning("onnx package not installed; skipping validation. Run: pip install onnx")
    except Exception as exc:
        raise RuntimeError(f"ONNX model validation failed: {exc}") from exc


def _try_simplify(path: str) -> None:
    """Attempt onnxsim simplification, logging a warning on failure."""
    try:
        import onnx  # type: ignore[import]
        import onnxsim  # type: ignore[import]

        model_simp, ok = onnxsim.simplify(path)
        if ok:
            onnx.save(model_simp, path)
            log.info("ONNX graph simplified successfully.")
        else:
            log.warning("onnxsim returned ok=False; keeping original graph.")
    except ImportError:
        log.warning("onnxsim not installed; skipping simplification. Run: pip install onnxsim")
    except Exception as exc:
        log.warning("onnxsim simplification failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a WSmart-Route encoder/policy to ONNX for Netron inspection.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint file containing the model.",
    )
    parser.add_argument(
        "--output_dir",
        default="assets/onnx",
        help="Directory for output .onnx files. Default: assets/onnx",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=50,
        help="Number of graph nodes in dummy input. Default: 50",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        help="Node embedding dimension. Default: 128",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dummy batch size (1 is sufficient for Netron). Default: 1",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxsim graph simplification after export.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ONNX export output.",
    )
    return parser


def main() -> None:
    """CLI entry point for exporting WSmart-Route models to ONNX."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Handle Lightning checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        log.warning(
            "Lightning checkpoint detected. To export a full policy, "
            "instantiate the model programmatically and call "
            "export_policy_components() directly."
        )
        keys = list(ckpt["state_dict"].keys())
        log.info("State-dict keys (first 10): %s", keys[:10])
        return

    model: nn.Module = ckpt if isinstance(ckpt, nn.Module) else ckpt.get("model")  # type: ignore[assignment]
    if model is None or not isinstance(model, nn.Module):
        raise ValueError(
            "Checkpoint does not contain a valid nn.Module. Use programmatic export via export_policy_components()."
        )

    paths = export_policy_components(
        policy=model,
        export_dir=args.output_dir,
        n_nodes=args.num_loc,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        opset_version=args.opset,
        simplify=args.simplify,
    )

    print("\nExported components:")
    for component, path in paths.items():
        print(f"  {component:<12} → {path}")


if __name__ == "__main__":
    main()
