"""
Batch runner — CLI entry point for the WSmart-Route Batch Manager.

Invoked by ``just batch-run`` or directly::

    uv run python -m logic.controllers.manager.batch_runner \
        --batch_cfg logic/configs/my_experiment.yaml

Or through ``main.py`` (future integration)::

    python main.py batch_run batch_cfg=logic/configs/my_experiment.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="batch_runner",
        description="WSmart-Route Batch Manager — run multiple experiments in sequence.",
    )
    parser.add_argument(
        "--batch_cfg",
        required=True,
        metavar="PATH",
        help="Path to the batch YAML configuration file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        default=False,
        help="Abort on the first job failure (overrides config).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the batch runner.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = at least one failure).
    """
    args = _parse_args(argv)

    from logic.controllers.manager import BatchManager

    mgr = BatchManager.from_yaml(args.batch_cfg)

    # CLI flags override YAML settings
    if args.dry_run:
        mgr._cfg["dry_run"] = True  # type: ignore[index]
    if args.fail_fast:
        mgr._cfg["fail_fast"] = True  # type: ignore[index]

    return mgr.run()


if __name__ == "__main__":
    sys.exit(main())
