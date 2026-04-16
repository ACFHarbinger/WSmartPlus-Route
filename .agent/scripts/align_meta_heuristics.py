#!/usr/bin/env python3
"""
Batch-patch meta-heuristic solvers to handle Tuple[bool, AcceptanceMetrics] from accept().

Pattern to find:
    accepted = criterion.accept(...)
    criterion.step(..., accepted)

Pattern to replace:
    accepted, _ = criterion.accept(...)
    criterion.step(..., accepted)
"""

import re
from pathlib import Path

# Targeted directories (absolute paths)
ROOT = Path("/home/pkhunter/Repositories/WSmart-Route")
DIRS = [
    ROOT / "logic/src/policies/route_construction/meta_heuristics",
    ROOT / "logic/src/policies/route_construction/matheuristics",
    ROOT / "logic/src/policies/route_construction/hyper_heuristics",
    ROOT / "logic/src/policies/route_construction/learning_heuristic_algorithms",
    ROOT / "logic/src/policies/helpers/local_search",
]


def patch_accept_calls(src: str) -> str:
    """
    Finds calls to .accept(...) and ensures they unpack a tuple.
    Handles:
        accepted = criterion.accept(...)
        if criterion.accept(...):
    """
    # 1. Assignment pattern: accepted = crit.accept(...)
    # We use a broad match for the variable name (accepted, is_accepted, etc.)
    # Note: we avoid matching if it already has a comma (already patched)
    src = re.sub(r"(?<!,)(?<!\w)(\w+)\s*=\s*([^ \n]+)\.accept\(", r"\1, _ = \2.accept(", src)

    # 2. Inline pattern: if crit.accept(...):
    # Pattern: if criterion.accept(...): -> if criterion.accept(...)[0]:
    # We avoid matching if [0] is already there.
    src = re.sub(r"if\s+([^ \n]+)\.accept\((.*?)\)(?!\[0\]):", r"if \1.accept(\2)[0]:", src)

    # 3. Cleanup: If we accidentally double-unpacked (e.g. accepted, _, _ = ...)
    src = src.replace(", _, _ =", ", _ =")

    return src


def process_file(path: Path) -> None:
    if path.suffix != ".py":
        return
    orig = path.read_text(encoding="utf-8")

    # We only care about files that actually use acceptance criteria
    if ".accept(" not in orig:
        return

    src = patch_accept_calls(orig)

    if src != orig:
        path.write_text(src, encoding="utf-8")
        print(f"  aligned: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    print("Aligning meta-heuristic callers...")
    for d in DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            process_file(p)
    print("Done.")
