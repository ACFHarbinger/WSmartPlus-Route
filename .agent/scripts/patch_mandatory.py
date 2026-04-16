#!/usr/bin/env python3
"""
Batch-patch mandatory selection strategies to return Tuple[List[int], SearchContext].

For each strategy file this script:
  1. Changes `select_bins()` return type from `-> List[int]` to
     `-> Tuple[List[int], SearchContext]`.
  2. Wraps every `return <list_expr>` with a freshly initialised SearchContext.
  3. Updates imports.

Usage:
    python .agent/scripts/patch_mandatory.py
"""

import re
from pathlib import Path

BASE = Path("/home/pkhunter/Repositories/WSmart-Route/logic/src/policies/mandatory_selection")
FILES = [p for p in BASE.glob("*.py") if not p.name.startswith("_")]

SEARCH_CONTEXT_IMPORT = "from logic.src.policies.context.search_context import SearchContext\n"


def patch_imports(src: str) -> str:
    """Add Tuple and SearchContext imports if missing."""
    # Ensure Tuple in typing import
    src = re.sub(
        r"from typing import ([^\n]+)",
        lambda m: (m.group(0) if "Tuple" in m.group(1) else f"from typing import {m.group(1).rstrip()}, Tuple"),
        src,
        count=1,
    )
    # Add SearchContext import after the typing import
    if "SearchContext" not in src:
        src = re.sub(
            r"(from typing import [^\n]+\n)",
            r"\1" + SEARCH_CONTEXT_IMPORT,
            src,
            count=1,
        )
    return src


def patch_select_signature(src: str) -> str:
    """Change 'def select_bins(...) -> List[int]:' to return Tuple."""
    src = re.sub(
        r"(def select_bins\(.*?\))\s*->\s*List\[int\]\s*:",
        r"\1 -> Tuple[List[int], SearchContext]:",
        src,
        flags=re.DOTALL,
    )
    return src


def get_class_name(src: str) -> str:
    m = re.search(r"^class (\w+)", src, re.MULTILINE)
    return m.group(1) if m else "UnknownSelection"


def patch_select_returns(src: str) -> str:  # noqa: C901
    """
    Wrap every `return <list_expr>` inside select_bins() with a SearchContext.
    """
    cls_name = get_class_name(src)
    lines = src.splitlines(keepends=True)
    out = []
    in_select = False
    select_indent: int = 0

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if re.match(r"def select_bins\(", stripped):
            in_select = True
            select_indent = indent
            out.append(line)
            continue

        if (
            in_select
            and stripped
            and not stripped.startswith("#")
            and indent <= select_indent
            and re.match(r"def |class ", stripped)
        ):
            in_select = False

        if in_select and stripped.startswith("return "):
            expr = stripped[len("return ") :].rstrip("\n")
            leading = line[: len(line) - len(line.lstrip())]
            ctx_init = f'SearchContext.initialize(selection_metrics={{"strategy": "{cls_name}"}})'
            out.append(f"{leading}return {expr}, {ctx_init}\n")
            continue

        out.append(line)

    return "".join(out)


def process_file(path: Path) -> None:
    orig = path.read_text(encoding="utf-8")
    src = patch_imports(orig)
    src = patch_select_signature(src)
    src = patch_select_returns(src)

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
