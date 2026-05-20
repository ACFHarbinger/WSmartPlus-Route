"""Utilities for removing targeted simulation runs from output files.

This module provides functions to surgically remove specific policy runs
from all simulation output artefacts:

- ``daily_{dist}_{N}N.json`` / ``log_mean_{N}N.json`` / ``log_std_{N}N.json``
  / ``log_full_{N}N.json`` — keyed JSON files
- ``log_realtime_{dist}_{N}N.jsonl`` — prefixed JSONL stream files
- ``checkpoints/`` — per-policy/sample ``.pkl`` files
- ``fill_history/{dist}/`` — per-policy XLSX files

Usage::

    from logic.src.utils.target.remover import PolicyFilter, remove_targeted_runs

    filt = PolicyFilter(
        distributions=["emp"],
        constructors=["alns"],
        improvers=["ftsp"],
    )
    removed = remove_targeted_runs(
        results_dir="assets/output/30_days/riomaior_100",
        policy_filter=filt,
        dry_run=True,      # preview only
    )
    for r in removed:
        print(r)

Attributes:
    remove_targeted_runs: High-level entry point.
    remove_from_json_file: Remove matching keys from a JSON results file.
    remove_from_jsonl_file: Remove matching log lines from a JSONL stream.
    remove_checkpoint_files: Delete matching checkpoint ``.pkl`` files.
    remove_fill_history_files: Delete matching XLSX fill-history files.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from logic.src.utils.target.matcher import (
    PolicyFilter,
    display_name_matches_filter,
    slug_matches_filter,
)

__all__ = [
    "PolicyFilter",
    "remove_targeted_runs",
    "remove_from_json_file",
    "remove_from_jsonl_file",
    "remove_checkpoint_files",
    "remove_fill_history_files",
]

# ---------------------------------------------------------------------------
# JSON files  (daily_*, log_mean_*, log_std_*, log_full_*)
# ---------------------------------------------------------------------------


def remove_from_json_file(
    path: str,
    policy_filter: PolicyFilter,
    dry_run: bool = False,
) -> List[str]:
    """Remove entries whose key matches *policy_filter* from a keyed JSON file.

    The file is re-written in-place with matching keys omitted.

    Args:
        path: Absolute or relative path to the JSON file.
        policy_filter: Filter criteria.  Only matching keys are removed.
        dry_run: When ``True`` the file is **not** modified; only the list of
            keys that *would* be removed is returned.

    Returns:
        List of removed (or would-be-removed) key strings.
    """
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as fh:
        try:
            data: dict = json.load(fh)
        except json.JSONDecodeError:
            return []

    removed: List[str] = []
    kept: dict = {}
    for key, value in data.items():
        if slug_matches_filter(key, policy_filter):
            removed.append(key)
        else:
            kept[key] = value

    if removed and not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(kept, fh, indent=1)

    return removed


# ---------------------------------------------------------------------------
# JSONL files  (log_realtime_*)
# The format is one line per day-log entry; each line is prefixed:
#   GUI_DAY_LOG_START:<Display Name>,<sample_id>,<day>,{...json...}
# ---------------------------------------------------------------------------

_JSONL_PREFIX_RE = re.compile(r"^GUI_DAY_LOG_START:([^,]+),")


def remove_from_jsonl_file(
    path: str,
    policy_filter: PolicyFilter,
    dry_run: bool = False,
) -> List[str]:
    """Remove lines whose display-name prefix matches *policy_filter* from a JSONL file.

    The file is re-written in-place with matching lines omitted.

    Args:
        path: Absolute or relative path to the ``.jsonl`` file.
        policy_filter: Filter criteria.
        dry_run: When ``True`` the file is **not** modified.

    Returns:
        Sorted unique list of display names that were (or would be) removed.
    """
    if not os.path.isfile(path):
        return []

    kept_lines: List[str] = []
    removed_names: set = set()

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = _JSONL_PREFIX_RE.match(line)
            if m:
                display_name = m.group(1).strip()
                if display_name_matches_filter(display_name, policy_filter):
                    removed_names.add(display_name)
                    continue
            kept_lines.append(line)

    if removed_names and not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(kept_lines)

    return sorted(removed_names)


# ---------------------------------------------------------------------------
# Checkpoints  (checkpoints/checkpoint_{slug}_{sample_id}_day{n}.pkl)
# ---------------------------------------------------------------------------


def remove_checkpoint_files(
    checkpoints_dir: str,
    policy_filter: PolicyFilter,
    dry_run: bool = False,
) -> List[str]:
    """Delete checkpoint ``.pkl`` files that match *policy_filter*.

    Filename pattern: ``checkpoint_{policy_slug}_{sample_id}_day{n}.pkl``

    Args:
        checkpoints_dir: Path to the ``checkpoints/`` directory.
        policy_filter: Filter criteria.
        dry_run: When ``True`` files are **not** deleted.

    Returns:
        List of deleted (or would-be-deleted) file paths.
    """
    removed: List[str] = []
    if not os.path.isdir(checkpoints_dir):
        return removed

    for fname in os.listdir(checkpoints_dir):
        if not fname.endswith(".pkl"):
            continue
        # Strip "checkpoint_" prefix and "_day{n}.pkl" suffix to get the slug
        stem = fname[len("checkpoint_") :] if fname.startswith("checkpoint_") else fname
        # Remove trailing _day{n}
        stem = re.sub(r"_day\d+$", "", stem)
        # Remove trailing _sample or _<digit> (sample id)
        stem = re.sub(r"_\d+$", "", stem)

        if slug_matches_filter(stem, policy_filter):
            fpath = os.path.join(checkpoints_dir, fname)
            removed.append(fpath)
            if not dry_run:
                os.remove(fpath)

    return removed


# ---------------------------------------------------------------------------
# Fill history  (fill_history/{dist}/{slug}{seed}_sample{id}.xlsx)
# ---------------------------------------------------------------------------


def remove_fill_history_files(
    fill_history_dir: str,
    policy_filter: PolicyFilter,
    dry_run: bool = False,
) -> List[str]:
    """Delete fill-history XLSX files that match *policy_filter*.

    Filename pattern: ``{policy_slug}{seed}_sample{id}.xlsx``
    The directory contains sub-directories named after distributions
    (``emp/``, ``gamma3/``, etc.).

    Args:
        fill_history_dir: Path to the ``fill_history/`` directory.
        policy_filter: Filter criteria.
        dry_run: When ``True`` files are **not** deleted.

    Returns:
        List of deleted (or would-be-deleted) file paths.
    """
    removed: List[str] = []
    if not os.path.isdir(fill_history_dir):
        return removed

    for dist_dir in os.listdir(fill_history_dir):
        dist_path = os.path.join(fill_history_dir, dist_dir)
        if not os.path.isdir(dist_path):
            continue

        # If distribution filter is set, skip sub-dirs that don't match
        if policy_filter.distributions:
            # dist_dir itself is the distribution name
            if not any(
                d.lower() == dist_dir.lower() or d.lower() in dist_dir.lower() for d in policy_filter.distributions
            ):
                continue

        for fname in os.listdir(dist_path):
            if not fname.endswith(".xlsx"):
                continue
            # Strip seed + sample suffix: e.g. "lookahead_aco_hh_custom_ftsp_emp42_sample0"
            stem = fname[: -len(".xlsx")]
            # Remove trailing "_sample<id>"
            stem = re.sub(r"_sample\d+$", "", stem)
            # Remove trailing seed digits (integer suffix after distribution tag)
            # Pattern: slug ends with _{dist}{seed} e.g. "_emp42"
            stem = re.sub(r"(\d+)$", "", stem).rstrip("_")

            if slug_matches_filter(stem, policy_filter):
                fpath = os.path.join(dist_path, fname)
                removed.append(fpath)
                if not dry_run:
                    os.remove(fpath)

    return removed


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def remove_targeted_runs(
    results_dir: str,
    policy_filter: PolicyFilter,
    distributions: Optional[List[str]] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Remove all artefacts for simulation runs matching *policy_filter*.

    Scans the given *results_dir* (e.g. ``assets/output/30_days/riomaior_100``)
    and removes matching entries from:

    - All ``*.json`` result files (daily, log_mean, log_std, log_full)
    - All ``*.jsonl`` realtime log files
    - All matching ``checkpoints/*.pkl`` files
    - All matching ``fill_history/**/*.xlsx`` files

    Args:
        results_dir: Path to the simulation results directory.
        policy_filter: Criteria specifying which runs to remove.
        distributions: Optional override for ``policy_filter.distributions``.
            Merged into the filter if provided.
        dry_run: When ``True`` no files are modified; the list of *would-be*
            removed items is returned for inspection.
        verbose: Print a summary of removed artefacts to stdout.

    Returns:
        List of all removed (or would-be-removed) item descriptions.
    """
    if distributions:
        policy_filter.distributions = list(set(policy_filter.distributions + distributions))

    all_removed: List[str] = []

    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        if verbose:
            print(f"[WARN] Results directory not found: {results_dir}")
        return all_removed

    # --- JSON result files ---
    for fpath in glob.glob(os.path.join(results_dir, "*.json")):
        removed_keys = remove_from_json_file(fpath, policy_filter, dry_run=dry_run)
        for key in removed_keys:
            desc = f"JSON key '{key}' from {os.path.basename(fpath)}"
            all_removed.append(desc)
            if verbose:
                prefix = "[DRY RUN]" if dry_run else "[REMOVED]"
                print(f"  {prefix} {desc}")

    # --- JSONL realtime logs ---
    for fpath in glob.glob(os.path.join(results_dir, "*.jsonl")):
        removed_names = remove_from_jsonl_file(fpath, policy_filter, dry_run=dry_run)
        for name in removed_names:
            desc = f"JSONL lines for '{name}' from {os.path.basename(fpath)}"
            all_removed.append(desc)
            if verbose:
                prefix = "[DRY RUN]" if dry_run else "[REMOVED]"
                print(f"  {prefix} {desc}")

    # --- Checkpoints ---
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    removed_files = remove_checkpoint_files(ckpt_dir, policy_filter, dry_run=dry_run)
    for fpath in removed_files:
        desc = f"checkpoint file {os.path.relpath(fpath, results_dir)}"
        all_removed.append(desc)
        if verbose:
            prefix = "[DRY RUN]" if dry_run else "[REMOVED]"
            print(f"  {prefix} {desc}")

    # --- Fill history ---
    fh_dir = os.path.join(results_dir, "fill_history")
    removed_files = remove_fill_history_files(fh_dir, policy_filter, dry_run=dry_run)
    for fpath in removed_files:
        desc = f"fill_history file {os.path.relpath(fpath, results_dir)}"
        all_removed.append(desc)
        if verbose:
            prefix = "[DRY RUN]" if dry_run else "[REMOVED]"
            print(f"  {prefix} {desc}")

    if verbose:
        status = "Would remove" if dry_run else "Removed"
        print(f"\n{status} {len(all_removed)} artefact(s) from {results_dir}")

    return all_removed
