#!/usr/bin/env python3
"""
Batch-patch acceptance criteria to return Tuple[bool, AcceptanceMetrics].

For each criterion file this script:
  1. Changes the `accept()` signature return type from `-> bool` to
     `-> Tuple[bool, AcceptanceMetrics]`.
  2. Wraps the existing `return <bool_expr>` statements in
     `result = <bool_expr>; return result, {<state>}`.
  3. Updates imports (adds `Tuple`, adds `AcceptanceMetrics` import).

The transformation is text-based (AST-free) because the existing files
follow a very consistent pattern.

Usage:
    python /tmp/patch_criteria.py
"""

import re
from pathlib import Path

BASE = Path("/home/pkhunter/Repositories/WSmart-Route/logic/src/policies/route_construction/acceptance_criteria")

# Files to patch (all except __init__ and base/)
FILES = [p for p in BASE.glob("*.py") if not p.name.startswith("_")]

ACCEPTANCE_METRICS_IMPORT = "from logic.src.policies.context.search_context import AcceptanceMetrics\n"


def patch_imports(src: str, filename: str) -> str:
    """Add Tuple and AcceptanceMetrics imports if missing."""
    # Ensure Tuple is in the typing import
    src = re.sub(
        r"from typing import ([^\n]+)",
        lambda m: (m.group(0) if "Tuple" in m.group(1) else f"from typing import {m.group(1).rstrip()}, Tuple"),
        src,
        count=1,
    )
    # Add AcceptanceMetrics import after the typing import line
    if "AcceptanceMetrics" not in src:
        src = re.sub(
            r"(from typing import [^\n]+\n)",
            r"\1" + ACCEPTANCE_METRICS_IMPORT,
            src,
            count=1,
        )
    return src


def patch_accept_signature(src: str) -> str:
    """Change 'def accept(...) -> bool:' to '-> Tuple[bool, AcceptanceMetrics]:'."""
    src = re.sub(
        r"(def accept\(.*?\))\s*->\s*bool\s*:",
        r"\1 -> Tuple[bool, AcceptanceMetrics]:",
        src,
        flags=re.DOTALL,
    )
    return src


def get_state_body(src: str) -> str:
    """Extract the dict literal from get_state() as a string, e.g. '{"temperature": self.T}'."""
    m = re.search(
        r"def get_state\(self\)\s*->\s*Dict\[str,\s*Any\]:\s*\n\s+return\s+(\{[^}]*\})",
        src,
        re.DOTALL,
    )
    if m:
        # Collapse multiline dict to single line for inline use
        raw = m.group(1)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw
    return "{}"


def patch_accept_body(src: str, state_dict: str) -> str:  # noqa: C901
    """
    Rewrite every `return <expr>` inside the accept() method body.

    Strategy:
      - Locate the accept() method by its signature.
      - Find each `return` statement within it.
      - Wrap: `return True` → `return True, {**state, "accepted": True, "delta": delta}`
      - Single-boolean returns are wrapped directly.
      - We use a simple line-by-line scanner bounded by indentation level.
    """
    lines = src.splitlines(keepends=True)
    out = []
    in_accept = False
    accept_indent: int = 0

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Detect accept() method start
        if re.match(r"def accept\(", stripped):
            in_accept = True
            accept_indent = indent
            out.append(line)
            continue

        # Detect method exit (new def at same/lower indent)
        if (
            in_accept
            and stripped
            and not stripped.startswith("#")
            and indent <= accept_indent
            and re.match(r"def |class ", stripped)
        ):
            in_accept = False

        if in_accept and stripped.startswith("return "):
            # Extract the expression after "return "
            expr = stripped[len("return ") :].rstrip("\n")
            leading = line[: len(line) - len(line.lstrip())]
            # Build metrics dict merging state_dict with runtime delta info
            if state_dict == "{}":
                metrics = '{"accepted": _accepted, "delta": candidate_obj - current_obj}'
            else:
                # Merge: {**<state>, "accepted": ..., "delta": ...}
                inner = state_dict[1:-1].strip()
                metrics = '{"accepted": _accepted, "delta": candidate_obj - current_obj, ' + inner + "}"

            new_lines = f"{leading}_accepted = bool({expr})\n{leading}return _accepted, {metrics}\n"
            out.append(new_lines)
            continue

        out.append(line)

    return "".join(out)


def process_file(path: Path) -> None:
    orig = path.read_text(encoding="utf-8")

    state_dict = get_state_body(orig)
    src = patch_imports(orig, path.name)
    src = patch_accept_signature(src)
    src = patch_accept_body(src, state_dict)

    # Only write if actually changed
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
