#!/usr/bin/env python3
"""
Batch-patch route improvers to return Tuple[List[int], ImprovementMetrics].

For each improver file this script:
  1. Changes the `process()` return type from `-> List[int]` to
     `-> Tuple[List[int], ImprovementMetrics]`.
  2. Wraps every `return <tour_expr>` with a metrics dict.
  3. Updates imports.

Usage:
    python .agent/scripts/patch_improvers.py
"""

import re
from pathlib import Path

BASE = Path("/home/pkhunter/Repositories/WSmart-Route/logic/src/policies/route_improvement")

# Files to patch — exclude base/, common/, and the two that need manual SA refactoring.
SKIP = {"__init__.py", "simulated_annealing.py", "ruin_recreate.py"}
FILES = [p for p in BASE.glob("*.py") if p.name not in SKIP and not p.name.startswith("_")]

IMPROVEMENT_METRICS_IMPORT = "from logic.src.policies.context.search_context import ImprovementMetrics\n"


def patch_imports(src: str) -> str:
    """Add Tuple and ImprovementMetrics imports if missing."""
    # Ensure Tuple in typing import
    src = re.sub(
        r"from typing import ([^\n]+)",
        lambda m: (m.group(0) if "Tuple" in m.group(1) else f"from typing import {m.group(1).rstrip()}, Tuple"),
        src,
        count=1,
    )
    # Add ImprovementMetrics import after the typing import line
    if "ImprovementMetrics" not in src:
        src = re.sub(
            r"(from typing import [^\n]+\n)",
            r"\1" + IMPROVEMENT_METRICS_IMPORT,
            src,
            count=1,
        )
    return src


def patch_process_signature(src: str) -> str:
    """Change 'def process(...) -> List[int]:' to Tuple[List[int], ImprovementMetrics]."""
    src = re.sub(
        r"(def process\(.*?\))\s*->\s*List\[int\]\s*:",
        r"\1 -> Tuple[List[int], ImprovementMetrics]:",
        src,
        flags=re.DOTALL,
    )
    return src


def patch_process_returns(src: str) -> str:  # noqa: C901
    """
    Wrap every `return <expr>` inside process() with an ImprovementMetrics dict.

    Strategy: line-by-line scan bounded by method indentation.
    We wrap `return tour_expr` → `return tour_expr, {"algorithm": cls_name}`.
    """
    # Extract class name for use in metrics
    cls_m = re.search(r"^class (\w+)", src, re.MULTILINE)
    cls_name = cls_m.group(1) if cls_m else "UnknownImprover"

    lines = src.splitlines(keepends=True)
    out = []
    in_process = False
    process_indent: int = 0

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Detect process() method start (not inside another method)
        if re.match(r"def process\(", stripped):
            in_process = True
            process_indent = indent
            out.append(line)
            continue

        # Detect method exit
        if (
            in_process
            and stripped
            and not stripped.startswith("#")
            and indent <= process_indent
            and re.match(r"def |class ", stripped)
        ):
            in_process = False

        if in_process and stripped.startswith("return "):
            expr = stripped[len("return ") :].rstrip("\n")
            leading = line[: len(line) - len(line.lstrip())]
            # Avoid double-wrapping already-tuple returns
            if expr.startswith("(") and "ImprovementMetrics" in src:
                out.append(line)
                continue
            metrics = f'{{"algorithm": "{cls_name}"}}'
            out.append(f"{leading}return {expr}, {metrics}\n")
            continue

        out.append(line)

    return "".join(out)


def process_file(path: Path) -> None:
    orig = path.read_text(encoding="utf-8")
    src = patch_imports(orig)
    src = patch_process_signature(src)
    src = patch_process_returns(src)

    if src != orig:
        path.write_text(src, encoding="utf-8")
        print(f"  patched: {path.name}")
    else:
        print(f"  skipped (no change): {path.name}")


if __name__ == "__main__":
    print(f"Processing {len(FILES)} files...")
    for f in sorted(FILES):
        process_file(f)
    print("Done.")
