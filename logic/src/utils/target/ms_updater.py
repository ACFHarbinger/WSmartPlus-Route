"""Policy YAML mandatory-selection config updater.

Batch-updates the ``mandatory_selection`` override in policy YAML files for
one or more route constructors.

Quick start::

    from logic.src.utils.target.ms_updater import update_mandatory_selection

    update_mandatory_selection(
        constructors=["aco_hh", "alns", "bpc"],
        ms_yaml="ms_service_level",
        keys=["service_level1", "service_level2"],
    )

CLI usage::

    python -m logic.src.cli.target_parser ms \\
        --constructors aco_hh alns bpc \\
        --file ms_service_level \\
        --keys service_level1 service_level2

Attributes:
    update_mandatory_selection: Update mandatory_selection in policy YAML files.
    list_available_ms_strategies: List available mandatory-selection strategy file stems.
    list_strategy_keys: List available keys within a strategy file.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIGS_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../../../configs/policies")
)

_MS_FIELD_RE = re.compile(r"(mandatory_selection:\s*)\{[^}]+\}")


def _resolve_stem(yaml_arg: str) -> str:
    """Normalise a yaml filename argument to a bare stem.

    Args:
        yaml_arg: e.g. ``"ms_service_level"``, ``"ms_service_level.yaml"``,
            or ``"other/ms_service_level.yaml"``.

    Returns:
        Bare stem, e.g. ``"ms_service_level"``.
    """
    name = os.path.basename(yaml_arg)
    if name.endswith(".yaml"):
        name = name[:-5]
    return name


def list_available_ms_strategies(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return sorted list of available mandatory-selection strategy file stems.

    Args:
        configs_dir: Path to the ``logic/configs/policies/`` directory.

    Returns:
        Sorted list of file stems, e.g. ``["ms_last_minute", "ms_service_level", ...]``.
    """
    other_dir = os.path.join(configs_dir, "other")
    if not os.path.isdir(other_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(other_dir)
        if f.startswith("ms_") and f.endswith(".yaml")
    )


def list_strategy_keys(ms_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return top-level keys defined in *ms_yaml*.

    These are the valid values for the ``--keys`` argument.

    Args:
        ms_yaml: Bare stem or filename of the strategy file.
        configs_dir: Path to the ``logic/configs/policies/`` directory.

    Returns:
        List of top-level YAML keys found in the file.
    """
    stem = _resolve_stem(ms_yaml)
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


def update_mandatory_selection(
    constructors: List[str],
    ms_yaml: str,
    keys: List[str],
    configs_dir: str = _DEFAULT_CONFIGS_DIR,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Tuple[str, int]]:
    """Update ``mandatory_selection`` in policy YAML files for *constructors*.

    Replaces every ``mandatory_selection: { ... }`` entry in the matching
    policy files with a new inline-dict value referencing *ms_yaml* and the
    given *keys*.

    Args:
        constructors: Route constructor names, e.g. ``["aco_hh", "alns", "bpc"]``.
        ms_yaml: Bare stem of the strategy file, e.g. ``"ms_service_level"``.
        keys: Keys to select from the strategy file, e.g. ``["service_level1"]``.
        configs_dir: Path to the ``logic/configs/policies/`` directory.
        dry_run: When ``True`` no files are modified.
        verbose: Print per-file update summary.

    Returns:
        List of ``(filepath, replacement_count)`` tuples for each modified file.

    Raises:
        FileNotFoundError: If the strategy file does not exist.
        ValueError: If a requested key is not present in the strategy file.
    """
    stem = _resolve_stem(ms_yaml)
    ms_file_path = os.path.join(configs_dir, "other", f"{stem}.yaml")
    if not os.path.isfile(ms_file_path):
        raise FileNotFoundError(
            f"Mandatory-selection file not found: {ms_file_path}\n"
            f"Available: {list_available_ms_strategies(configs_dir)}"
        )

    available_keys = list_strategy_keys(stem, configs_dir)
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

        new_content, count = _MS_FIELD_RE.subn(replacement, content)

        if count == 0:
            if verbose:
                print(f"  [SKIP] No mandatory_selection {{...}} found in policy_{constructor}.yaml")
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
