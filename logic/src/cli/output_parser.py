"""CLI parser for removing targeted simulation runs.

Integrates with the WSmart+ Route CLI registry as the ``clean_results``
sub-command, and can also be run standalone::

    # Via main CLI
    python main.py clean_results \\
        --results-dir assets/output/30_days/riomaior_100 \\
        --distribution emp \\
        --constructor alns \\
        --improver ftsp \\
        --dry-run

    # Standalone
    uv run python -m logic.src.cli.output_parser \\
        --results-dir assets/output/30_days/riomaior_100 \\
        --ms-strategy lookahead --distribution gamma3

All filter options are optional; an omitted option matches *any* value.
Multiple values can be passed to the same option (space-separated).

Attributes:
    add_output_args: Register arguments on an existing sub-parser.
    main: Standalone CLI entry point.
"""

from __future__ import annotations

import argparse
import sys

from logic.src.utils.output.remover import PolicyFilter, remove_targeted_runs


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register ``clean_results`` arguments on *parser*.

    Args:
        parser: An :class:`argparse.ArgumentParser` (or sub-parser) to populate.
    """
    parser.add_argument(
        "--results-dir",
        required=True,
        metavar="DIR",
        help=(
            "Path to the results directory, "
            "e.g. 'assets/output/30_days/riomaior_100'.  "
            "Supports any area / day-count combination."
        ),
    )
    parser.add_argument(
        "--distribution",
        nargs="+",
        default=[],
        metavar="DIST",
        help="Distribution(s) to target: emp  gamma1  gamma2  gamma3 …",
    )
    parser.add_argument(
        "--constructor",
        nargs="+",
        default=[],
        metavar="CON",
        help="Route constructor(s) to target: alns  hgs  aco_hh  bpc  pg_clns  sans  psoma …",
    )
    parser.add_argument(
        "--ms-strategy",
        nargs="+",
        default=[],
        metavar="MS",
        help="Mandatory-selection strategy/ies: lookahead  last_minute  regular …",
    )
    parser.add_argument(
        "--improver",
        nargs="+",
        default=[],
        metavar="IMP",
        help="Route improver(s): ftsp  fast_tsp  rls  rds  none …",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact token matching instead of the default substring matching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without modifying any files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-item output; only print the final count.",
    )


def _run_from_namespace(args: argparse.Namespace) -> int:
    """Execute the clean-results action from parsed *args*.

    Args:
        args: Parsed namespace produced by :func:`add_output_args`.

    Returns:
        Exit code (0 = success / items removed, 1 = nothing matched or error).
    """
    if not any([args.distribution, args.constructor, args.ms_strategy, args.improver]):
        print(
            "[ERROR] At least one filter option must be provided "
            "(--distribution, --constructor, --ms-strategy, --improver).\n"
            "Run with --help for usage.",
            file=sys.stderr,
        )
        return 1

    policy_filter = PolicyFilter(
        distributions=args.distribution,
        constructors=args.constructor,
        ms_strategies=args.ms_strategy,
        improvers=args.improver,
        exact_match=args.exact,
    )

    removed = remove_targeted_runs(
        results_dir=args.results_dir,
        policy_filter=policy_filter,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if args.quiet:
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {len(removed)} artefact(s).")

    return 0 if removed else 1


def main(argv=None) -> int:
    """Standalone CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m logic.src.cli.output_parser",
        description=(
            "Remove targeted simulation runs from WSmart+ Route output artefacts.\n\n"
            "Each filter option is OR-matched within its category and AND-matched "
            "across categories.  Omitting a category matches everything."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_output_args(parser)
    args = parser.parse_args(argv)
    return _run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
