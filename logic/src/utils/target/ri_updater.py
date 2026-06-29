"""Policy YAML route-improvement config updater.

Batch-updates the ``route_improvement`` override in policy YAML files for
one or more route constructors.

Quick start::

    from logic.src.utils.target.ri_updater import update_route_improvement

    update_route_improvement(
        constructors=["aco_hh", "alns", "bpc"],
        ri_yaml="ri_ftsp",
        keys=["ftsp"],
    )

CLI usage::

    python -m logic.src.cli.target_parser ri \\
        --constructors aco_hh alns bpc \\
        --file ri_ftsp \\
        --keys ftsp

Attributes:
    update_route_improvement: Update route_improvement in policy YAML files.
    list_available_ri_improvers: List available route-improver file stems.
    list_improver_keys: List available keys within an improver file.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIGS_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../../../configs/policies")
)

_RI_FIELD_RE = re.compile(r"(route_improvement:\s*)\{[^}]+\}")


def _resolve_stem(yaml_arg: str) -> str:
    """Normalise a yaml filename argument to a bare stem.

    Args:
        yaml_arg: e.g. ``"ri_ftsp"``, ``"ri_ftsp.yaml"``,
            or ``"other/ri_ftsp.yaml"``.

    Returns:
        Bare stem, e.g. ``"ri_ftsp"``.
    """
    name = os.path.basename(yaml_arg)
    if name.endswith(".yaml"):
        name = name[:-5]
    return name


def list_available_ri_improvers(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return sorted list of available route-improver file stems.

    Args:
        configs_dir: Path to the ``logic/configs/policies/`` directory.

    Returns:
        Sorted list of file stems, e.g. ``["ri_cls", "ri_ftsp", "ri_rls", ...]``.
    """
    other_dir = os.path.join(configs_dir, "other")
    if not os.path.isdir(other_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(other_dir)
        if f.startswith("ri_") and f.endswith(".yaml")
    )


def list_improver_keys(ri_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return top-level keys defined in *ri_yaml*.

    These are the valid values for the ``--keys`` argument.

    Args:
        ri_yaml: Bare stem or filename of the improver file.
        configs_dir: Path to the ``logic/configs/policies/`` directory.

    Returns:
        List of top-level YAML keys found in the file.
    """
    stem = _resolve_stem(ri_yaml)
    path = os.path.join(configs_dir, "other", f"{stem}.yaml")
    if not os.path.isfile(path):
        return []
    keys: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip()
            if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", stripped)
                if m:
                    keys.append(m.group(1))
    return keys


def update_route_improvement(
    constructors: List[str],
    ri_yaml: str,
    keys: List[str],
    configs_dir: str = _DEFAULT_CONFIGS_DIR,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Tuple[str, int]]:
    """Update ``route_improvement`` in policy YAML files for *constructors*.

    Replaces every ``route_improvement: { ... }`` entry in the matching
    policy files with a new inline-dict value referencing *ri_yaml* and the
    given *keys*.  Entries using an empty list (``route_improvement: []``)
    are intentionally left unchanged.

    Args:
        constructors: Route constructor names, e.g. ``["aco_hh", "alns", "bpc"]``.
        ri_yaml: Bare stem of the improver file, e.g. ``"ri_ftsp"``.
        keys: Keys to select from the improver file, e.g. ``["ftsp"]``.
        configs_dir: Path to the ``logic/configs/policies/`` directory.
        dry_run: When ``True`` no files are modified.
        verbose: Print per-file update summary.

    Returns:
        List of ``(filepath, replacement_count)`` tuples for each modified file.

    Raises:
        FileNotFoundError: If the improver file does not exist.
        ValueError: If a requested key is not present in the improver file.
    """
    stem = _resolve_stem(ri_yaml)
    ri_file_path = os.path.join(configs_dir, "other", f"{stem}.yaml")
    if not os.path.isfile(ri_file_path):
        raise FileNotFoundError(
            f"Route-improver file not found: {ri_file_path}\n"
            f"Available: {list_available_ri_improvers(configs_dir)}"
        )

    available_keys = list_improver_keys(stem, configs_dir)
    for key in keys:
        if key not in available_keys:
            raise ValueError(
                f"Key '{key}' not found in {stem}.yaml. "
                f"Available keys: {available_keys}"
            )

    keys_str = ", ".join(f'"{k}"' for k in keys)
    replacement = f'\\g<1>{{ "other/{stem}.yaml": [{keys_str}] }}'

    modified: List[Tuple[str, int]] = []

    for constructor in constructors:
        policy_file = os.path.join(configs_dir, f"policy_{constructor}.yaml")
        if not os.path.isfile(policy_file):
            if verbose:
                print(f"  [WARN] Policy file not found: {policy_file}")
            continue

        with open(policy_file, "r", encoding="utf-8") as fh:
            content = fh.read()

        new_content, count = _RI_FIELD_RE.subn(replacement, content)

        if count == 0:
            if verbose:
                print(f"  [SKIP] No route_improvement {{...}} found in policy_{constructor}.yaml")
            continue

        modified.append((policy_file, count))

        if verbose:
            prefix = "[DRY RUN]" if dry_run else "[UPDATED]"
            print(f"  {prefix} policy_{constructor}.yaml — {count} occurrence(s) → {stem}: {keys}")

        if not dry_run:
            with open(policy_file, "w", encoding="utf-8") as fh:
                fh.write(new_content)

    if verbose:
        status = "Would update" if dry_run else "Updated"
        print(f"\n{status} {len(modified)} policy file(s).")

    return modified
