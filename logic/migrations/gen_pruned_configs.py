#!/usr/bin/env python
"""
Migration script: generate pruned_config.yaml for existing simulation output directories.

Iterates over all assets/output/**/hydra/config.yaml files and generates a
pruned_config.yaml alongside them if one does not already exist.

The pruned config contains only the task-relevant sections (sim, tracking,
global fields, and selected policies) with mandatory_selection and
route_improvement YAML references expanded inline.

Usage:
    python logic/migrations/gen_pruned_configs.py
    python logic/migrations/gen_pruned_configs.py --force       # overwrite existing
    python logic/migrations/gen_pruned_configs.py --dry-run     # preview only
    python logic/migrations/gen_pruned_configs.py --output-dir assets/output/30days
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Reference expansion helpers (mirrors orchestrator logic)
# ---------------------------------------------------------------------------

def _expand_other_ref(ref_dict: dict, root_dir: str) -> dict:
    """Expand a {yaml_file: [key1, key2]} reference by loading the referenced YAML."""
    result = {}
    for yaml_path, keys in ref_dict.items():
        full_path = os.path.join(root_dir, "logic", "configs", "policies", yaml_path)
        if not os.path.exists(full_path):
            return ref_dict  # Keep original if file not found
        with open(full_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key in (keys or []):
            if key in yaml_data:
                result[key] = yaml_data[key]
    return result


def _expand_refs_recursive(obj: Any, root_dir: str) -> Any:
    """Recursively expand mandatory_selection and route_improvement YAML references."""
    _expandable = ("mandatory_selection", "route_improvement")
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in _expandable and isinstance(v, dict):
                result[k] = _expand_other_ref(v, root_dir)
            else:
                result[k] = _expand_refs_recursive(v, root_dir)
        return result
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            if isinstance(item, dict) and len(item) == 1:
                key = next(iter(item))
                val = item[key]
                if key in _expandable and isinstance(val, dict):
                    new_list.append({key: _expand_other_ref(val, root_dir)})
                else:
                    new_list.append({key: _expand_refs_recursive(val, root_dir)})
            elif isinstance(item, dict):
                new_list.append(_expand_refs_recursive(item, root_dir))
            else:
                new_list.append(item)
        return new_list
    else:
        return obj


# ---------------------------------------------------------------------------
# Pruned config generation
# ---------------------------------------------------------------------------

def generate_pruned_config(full: dict, root_dir: str) -> str:
    """Build and return a pruned YAML string from a full Hydra config dict."""
    sim = full.get("sim", {}) or {}
    active_policies = sim.get("policies", []) or []

    pruned: dict = {}
    for key in ("task", "seed", "device", "start", "run_name"):
        if key in full:
            pruned[key] = full[key]
    for section in ("sim", "tracking"):
        if section in full:
            pruned[section] = copy.deepcopy(full[section])

    p_full = full.get("p", {}) or {}
    p_pruned = {}
    for pol_name in active_policies:
        if pol_name in p_full:
            p_pruned[pol_name] = _expand_refs_recursive(
                copy.deepcopy(p_full[pol_name]), root_dir
            )
    pruned["p"] = p_pruned

    return yaml.dump(pruned, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Main migration logic
# ---------------------------------------------------------------------------

def find_config_files(output_dir: str):
    """Yield all hydra/config.yaml paths under output_dir."""
    for root, _dirs, files in os.walk(output_dir):
        if os.path.basename(root) == "hydra" and "config.yaml" in files:
            yield os.path.join(root, "config.yaml")


def migrate(output_dir: str, force: bool = False, dry_run: bool = False) -> None:
    root_dir = str(_REPO_ROOT)
    config_files = sorted(find_config_files(output_dir))

    if not config_files:
        print(f"No hydra/config.yaml files found under {output_dir}")
        return

    print(f"Found {len(config_files)} config file(s) under {output_dir}")
    skipped = generated = errors = 0

    for config_path in config_files:
        hydra_dir = os.path.dirname(config_path)
        pruned_path = os.path.join(hydra_dir, "pruned_config.yaml")

        if os.path.exists(pruned_path) and not force:
            skipped += 1
            continue

        rel = os.path.relpath(config_path, output_dir)
        try:
            with open(config_path) as f:
                full = yaml.safe_load(f) or {}

            pruned_yaml = generate_pruned_config(full, root_dir)

            if dry_run:
                print(f"[DRY RUN] Would write: {pruned_path}")
            else:
                with open(pruned_path, "w") as f:
                    f.write(pruned_yaml)
                print(f"  Generated: {rel} → pruned_config.yaml")
            generated += 1
        except Exception as e:
            print(f"  ERROR processing {rel}: {e}")
            errors += 1

    action = "Would generate" if dry_run else "Generated"
    print(
        f"\nDone. {action} {generated}, skipped {skipped} (already exist), {errors} error(s)."
    )
    if skipped and not force:
        print("Use --force to regenerate existing pruned_config.yaml files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pruned_config.yaml for existing runs")
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "assets" / "output"),
        help="Root directory to search for hydra/config.yaml files (default: assets/output)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing pruned_config.yaml files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files would be written without actually writing",
    )
    args = parser.parse_args()
    migrate(args.output_dir, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
