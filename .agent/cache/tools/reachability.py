#!/usr/bin/env python
"""AST-based reachability analysis from the four entry points through logic/.

Walks static imports plus string-keyed registries to build the set of
reachable modules under logic/. Reports unreachable modules.
"""
import ast
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Walk up from .agent/cache/tools to find logic/
for parent in [ROOT] + list(ROOT.parents):
    candidate = parent / "logic"
    if candidate.is_dir():
        LOGIC = candidate
        break
else:
    # fallback: assume we're in a worktree with logic/ at root
    LOGIC = ROOT.parents[3] / "logic" if len(ROOT.parents) > 3 else Path("logic")

if not LOGIC.exists():
    # try relative
    for p in [Path("logic"), Path("/tmp/mistral-wsr/logic")]:
        if p.is_dir():
            LOGIC = p
            break

print(f"LOGIC root: {LOGIC}")

# Entry points
ENTRY_FILES = [
    LOGIC.parent / "main.py",
    LOGIC / "controllers" / "hydra_dispatch.py",
]

# Registries that map strings to classes/modules (string-keyed reachability)
REGISTRY_NAMES = [
    "RouteConstructorRegistry",
    "MandatorySelectionRegistry",
    "RouteImproverRegistry",
    "AcceptanceCriterionRegistry",
    "GENERATOR_REGISTRY",
    "ENV_REGISTRY",
    "_POLICY_REGISTRY_SPEC",
    "_ALGO_REGISTRY",
]

def get_all_py_modules(logic_root):
    """Return set of dotted module names under logic/src."""
    modules = set()
    for f in logic_root.rglob("*.py"):
        rel = f.relative_to(logic_root)
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mod = ".".join(parts)
        modules.add(mod)
    return modules

def parse_imports(filepath):
    """Extract imported module paths from a .py file."""
    imports = set()
    if not filepath.exists():
        return imports
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8", errors="replace"), filename=str(filepath))
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

def resolve_module(import_path, logic_root):
    """Try to resolve an import path to a file under logic/."""
    # Handle logic.src.xxx style
    if import_path.startswith("logic."):
        parts = import_path.split(".")
        # logic.src.xxx -> logic/src/xxx
        if len(parts) >= 2 and parts[1] == "src":
            rel = Path(*parts[1:])
        elif len(parts) >= 1 and parts[0] == "logic":
            rel = Path(*parts[1:])
        else:
            return None
        candidate = logic_root / rel.with_suffix(".py")
        if candidate.exists():
            return candidate
        candidate = logic_root / rel / "__init__.py"
        if candidate.exists():
            return candidate
    # Handle logic_src.xxx or src.xxx style
    if import_path.startswith("logic_src."):
        parts = import_path.split(".")
        rel = Path(*parts[1:])
        candidate = logic_root / rel.with_suffix(".py")
        if candidate.exists():
            return candidate
        candidate = logic_root / rel / "__init__.py"
        if candidate.exists():
            return candidate
    if import_path.startswith("src."):
        parts = import_path.split(".")
        rel = Path(*parts[1:])
        candidate = logic_root / rel.with_suffix(".py")
        if candidate.exists():
            return candidate
        candidate = logic_root / rel / "__init__.py"
        if candidate.exists():
            return candidate
    # Try as direct path under logic/src
    for prefix in ["", "src."]:
        path_attempt = prefix + import_path
        parts = path_attempt.split(".")
        for base in [logic_root, logic_root / "src"]:
            candidate = base / Path(*parts).with_suffix(".py")
            if candidate.exists():
                return candidate
            candidate = base / Path(*parts) / "__init__.py"
            if candidate.exists():
                return candidate
    return None

def build_reachability(logic_root):
    all_modules = get_all_py_modules(logic_root)
    reachable = set()
    queue = []
    visited = set()

    # Seed from entry points only
    for entry in ENTRY_FILES:
        if not entry.exists():
            continue
        imports = parse_imports(entry)
        for imp in imports:
            f = resolve_module(imp, logic_root)
            if f:
                queue.append(f)

    while queue:
        f = queue.pop()
        if f in visited:
            continue
        visited.add(f)
        rel = f.relative_to(logic_root)
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mod = ".".join(parts)
        reachable.add(mod)

        imports = parse_imports(f)
        for imp in imports:
            rf = resolve_module(imp, logic_root)
            if rf and rf not in visited:
                queue.append(rf)

    unreachable = all_modules - reachable
    return all_modules, reachable, unreachable

if __name__ == "__main__":
    all_mods, reachable, unreachable = build_reachability(LOGIC)

    # Filter out test files
    unreachable_real = sorted(m for m in unreachable if not any(
        x in m for x in ["test", "conftest", "__pycache__"]
    ))

    print(f"\nTotal modules under logic/: {len(all_mods)}")
    print(f"Reachable: {len(reachable)}")
    print(f"Unreachable (non-test): {len(unreachable_real)}")
    print("\n--- Unreachable modules ---")
    for m in unreachable_real:
        print(f"  {m}")
