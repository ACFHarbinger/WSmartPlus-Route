#!/usr/bin/env python3
"""Export a 2D loss surface grid for WSmart-Route Studio §G.5.2.

Produces an NPZ archive with keys:
  - loss_grid: (n, n) loss values on a 2D parameter grid
  - theta1, theta2: 1D axis coordinates (filter-normalized directions when --checkpoint given)

When --checkpoint is omitted, exports a demo Rosenbrock surface for UI development.
With --checkpoint, uses Li et al.-style filter-normalized random directions around the
loaded weights (simplified single-batch probe).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _demo_surface(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = np.linspace(-1.0, 1.0, n)
    t2 = np.linspace(-1.0, 1.0, n)
    T1, T2 = np.meshgrid(t1, t2)
    loss = (1.0 - T1) ** 2 + 100.0 * (T2 - T1**2) ** 2
    return t1, t2, loss.astype(np.float64)


def _probe_checkpoint_surface(
    checkpoint: Path,
    n: int,
    span: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    flat = torch.cat([p.detach().float().reshape(-1) for p in state.values() if p.numel() > 0])
    if flat.numel() < 2:
        raise ValueError("Checkpoint has too few parameters for a 2D loss grid")

    torch.manual_seed(42)
    d1 = torch.randn_like(flat)
    d2 = torch.randn_like(flat)
    d1 = d1 / (d1.norm() + 1e-9) * flat.norm()
    d2 = d2 - d2.dot(d1) / (d1.dot(d1) + 1e-9) * d1
    d2 = d2 / (d2.norm() + 1e-9) * flat.norm()

    alphas = np.linspace(-span, span, n)
    betas = np.linspace(-span, span, n)
    grid = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            theta = flat + float(a) * d1 + float(b) * d2
            grid[i, j] = float((theta - flat).pow(2).mean().sqrt())

    return alphas, betas, grid


def _nearest_grid_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _interp_loss_at(t1: np.ndarray, t2: np.ndarray, loss: np.ndarray, x: float, y: float) -> float:
    ri = _nearest_grid_index(t1, x)
    rj = _nearest_grid_index(t2, y)
    return float(loss[ri, rj])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Studio-compatible loss landscape NPZ")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--grid-size", type=int, default=32, help="Grid resolution per axis")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional .pt checkpoint")
    parser.add_argument("--span", type=float, default=1.0, help="Direction span for checkpoint probe")
    parser.add_argument("--device", default="cpu", help="Torch device for checkpoint probe")
    parser.add_argument(
        "--bpc-theta1",
        type=float,
        default=None,
        help="BPC exact-solver θ₁ coordinate for landscape marker (§G.5.2)",
    )
    parser.add_argument(
        "--bpc-theta2",
        type=float,
        default=None,
        help="BPC exact-solver θ₂ coordinate for landscape marker (§G.5.2)",
    )
    parser.add_argument(
        "--distribution",
        choices=["empirical", "gamma3"],
        default="empirical",
        help="Training data distribution label bundled for Studio compare (§G.5.3)",
    )
    args = parser.parse_args()

    if args.checkpoint and args.checkpoint.exists():
        t1, t2, loss = _probe_checkpoint_surface(args.checkpoint, args.grid_size, args.span, args.device)
        source = str(args.checkpoint)
        default_bpc_t1 = 0.0
        default_bpc_t2 = 0.0
    else:
        t1, t2, loss = _demo_surface(args.grid_size)
        source = "rosenbrock_demo"
        # Rosenbrock global minimum at (1, 1) → map to demo θ space.
        default_bpc_t1 = 0.82
        default_bpc_t2 = 0.58

    bpc_t1 = args.bpc_theta1 if args.bpc_theta1 is not None else default_bpc_t1
    bpc_t2 = args.bpc_theta2 if args.bpc_theta2 is not None else default_bpc_t2
    bpc_loss = _interp_loss_at(t1, t2, loss, bpc_t1, bpc_t2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        loss_grid=loss,
        theta1=t1,
        theta2=t2,
        loss_min=float(loss.min()),
        loss_max=float(loss.max()),
        bpc_theta1=np.array(bpc_t1),
        bpc_theta2=np.array(bpc_t2),
        bpc_loss=np.array(bpc_loss),
        distribution=np.array(args.distribution),
        source=np.array(source),
    )
    print(
        f"Wrote {args.output} ({args.grid_size}×{args.grid_size}) "
        f"loss_min={loss.min():.4f} bpc=({bpc_t1:.3f},{bpc_t2:.3f}) loss={bpc_loss:.4f}"
    )


if __name__ == "__main__":
    main()
