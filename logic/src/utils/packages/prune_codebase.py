"""Pruning engine for WSmart-Route algorithm export pipeline.

Reads ``ci/export_config.json`` to enumerate every known algorithm acronym per
category, then deletes all algorithms that the caller did NOT request to keep.
Deletion is delegated to :func:`cleanup_helper.clean_by_acronym`, which already
handles YAML configs, Python dataclass configs, implementation directories /
files, and ``__init__.py`` / ``factory.py`` import cleanup.

Also handles:
- Optional-feature removal via ``--drop-features`` (META_LEARNING, HPO, EVAL,
  SECURITY, CALLBACKS, ENUMS, UI_LOGIC, DATA_WEB).
- Model-specific subnet pruning: decoders/, encoders/, and factories/ within
  ``logic/src/models/subnets/`` are trimmed to only the subdirectories required
  by the kept model set (driven by ``subnet_deps`` in the config).

Usage (standalone — normally called by packager.py)::

    python logic/src/utils/packages/prune_codebase.py \\
        --constructors HGS ALNS BPC \\
        --selectors MS_REGULAR MS_SAVINGS \\
        --improvement RI_OROPT RI_S2OPT \\
        --acceptance AC_GD AC_OI \\
        --models AM TAM \\
        --rl-algorithms RL_REINFORCE RL_PPO \\
        --drop-features META_LEARNING HPO SECURITY

Omitting a category flag entirely means "keep everything in that category".
Passing the flag with no values means "prune the entire category".
Omitting ``--drop-features`` keeps all optional features intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Path bootstrap — makes cleanup_helper importable when run directly
# ---------------------------------------------------------------------------
_PACKAGES_DIR = Path(__file__).resolve().parent
# parents[4] of the *file* == parents[3] of the directory; matches cleanup_helper.get_project_root()
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # WSmart-Route/
if str(_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGES_DIR))

from cleanup_helper import clean_by_acronym  # noqa: E402  (after sys.path patch)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONFIG_PATH = _PROJECT_ROOT / "ci" / "export_config.json"

# Maps the category keys used in the JSON to the --flag names in this CLI.
_CATEGORY_TO_FLAG: Dict[str, str] = {
    "constructor": "constructors",
    "selector": "selectors",
    "improvement": "improvement",
    "acceptance": "acceptance",
    "joint": "joint",
    "model": "models",
    "rl_algorithm": "rl_algorithms",
    "imitation_policy": "imitation_policies",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> Dict:
    if not _CONFIG_PATH.exists():
        _die(f"export_config.json not found at {_CONFIG_PATH}")
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _die(message: str) -> None:
    print(f"[prune_codebase] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _log(message: str) -> None:
    print(f"[prune_codebase] {message}")


def _collect_algorithms_by_category(config: Dict) -> Dict[str, Dict[str, str]]:
    """Return mapping: category_key → {USER_ACRONYM: internal_acronym}.

    Flattens all nested groups inside ``config["algorithms"]`` so that
    comment-only ``_comment*`` keys are silently skipped.
    """
    result: Dict[str, Dict[str, str]] = {cat: {} for cat in _CATEGORY_TO_FLAG}

    def _walk(node: Dict) -> None:
        for key, value in node.items():
            if key.startswith("_comment"):
                continue
            if not isinstance(value, dict):
                continue
            cat = value.get("category")
            internal = value.get("internal_acronym", "")
            if cat and cat in result and internal:
                result[cat][key.upper()] = internal
            else:
                # Nested group (e.g. "constructors", "selectors" wrapper objects)
                _walk(value)

    _walk(config.get("algorithms", {}))
    return result


def _build_cleanup_kwargs(category_name: str, config: Dict) -> Dict:
    """Extract the per-category cleanup kwargs from the JSON ``categories`` block."""
    cat_block = config["categories"].get(category_name)
    if not cat_block:
        _die(f"Category '{category_name}' not found in export_config.json categories block.")
    return {
        "yaml_dirs": cat_block.get("yaml_dirs", []),
        "config_dirs": cat_block.get("config_dirs", []),
        "impl_dirs": cat_block.get("impl_dirs", []),
        "yaml_prefixes": cat_block.get("yaml_prefixes"),
        "config_prefixes": cat_block.get("config_prefixes"),
    }


def _validate_acronyms(
    requested: Set[str],
    known: Set[str],
    category: str,
) -> None:
    """Warn about unknown acronyms so the caller can catch typos early."""
    unknown = requested - known
    if unknown:
        _log(
            f"WARNING: The following acronyms are not listed in export_config.json "
            f"for category '{category}' and will be ignored: "
            + ", ".join(sorted(unknown))
        )


def prune_optional_features(
    drop_features: List[str],
    config: Dict,
    dry_run: bool,
) -> None:
    """Run remove_*.py scripts for every feature in *drop_features*.

    Args:
        drop_features: Upper-cased feature keys to remove (e.g. ``["HPO", "SECURITY"]``).
        config: Parsed export config dict.
        dry_run: If True, log actions without executing scripts.
    """
    opt = config.get("optional_features", {})
    if not drop_features:
        return

    known_features = {k for k in opt if not k.startswith("_")}
    unknown = set(drop_features) - known_features
    if unknown:
        _log(
            f"WARNING: Unknown optional features (will be ignored): "
            + ", ".join(sorted(unknown))
        )

    for feature_key in sorted(drop_features):
        feature_key = feature_key.upper()
        if feature_key not in opt:
            continue
        feature = opt[feature_key]
        script_rel = feature.get("remove_script")
        if script_rel is None:
            _log(
                f"Optional feature '{feature_key}' has no remove_script — "
                "skipping (delete manually if needed)."
            )
            continue

        script_path = _PROJECT_ROOT / script_rel
        if not script_path.exists():
            _log(f"WARNING: remove script for '{feature_key}' not found at {script_path} — skipping.")
            continue

        if dry_run:
            _log(f"  [DRY-RUN] Would run remove script for feature '{feature_key}': {script_path}")
        else:
            import subprocess  # noqa: PLC0415
            _log(f"  Dropping optional feature '{feature_key}' via {script_path.name} …")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    _log(f"    {line}")
            if result.returncode != 0:
                _log(f"  ERROR: {script_path.name} exited with code {result.returncode}")
                if result.stderr:
                    for line in result.stderr.splitlines():
                        _log(f"    STDERR: {line}")
            else:
                _log(f"  Feature '{feature_key}' removed successfully.")


def prune_subnets(
    keep_model_acronyms: Optional[List[str]],
    config: Dict,
    dry_run: bool,
) -> None:
    """Prune model-specific subnet subdirectories for unkept models.

    When *keep_model_acronyms* is None all models are kept → nothing pruned.
    Otherwise, computes the union of ``subnet_deps`` for the kept models and
    removes encoder/decoder/factory subdirs not in that union.

    Always-keep entries (``common`` for encoders/decoders, ``attention.py`` and
    ``base.py`` for factories) are never removed.
    """
    if keep_model_acronyms is None:
        _log("Subnets: no model selection provided — keeping all subnets.")
        return

    pruning_cfg = config.get("subnet_pruning", {})
    models_cfg = config.get("algorithms", {}).get("models", {})

    # Collect union of required subnet deps across kept models.
    needed_decoders: Set[str] = set()
    needed_encoders: Set[str] = set()
    needed_factories: Set[str] = set()

    keep_upper = {a.upper() for a in keep_model_acronyms}

    for user_acr, model_entry in models_cfg.items():
        if user_acr.startswith("_comment"):
            continue
        if not isinstance(model_entry, dict):
            continue
        if user_acr.upper() not in keep_upper:
            continue
        deps = model_entry.get("subnet_deps", {})
        needed_decoders.update(deps.get("decoders", []))
        needed_encoders.update(deps.get("encoders", []))
        needed_factories.update(deps.get("factories", []))

    prunable_types = pruning_cfg.get("prunable_types", {})

    # --- Prune decoder subdirs ---
    decoder_cfg = prunable_types.get("decoders", {})
    decoder_base = _PROJECT_ROOT / decoder_cfg.get("base_path", "logic/src/models/subnets/decoders")
    always_keep_dec: Set[str] = set(decoder_cfg.get("always_keep", ["common"]))
    all_dec_subdirs: List[str] = decoder_cfg.get("all_subdirs", [])
    _prune_subdirs(decoder_base, all_dec_subdirs, needed_decoders | always_keep_dec, "decoder", dry_run)

    # --- Prune encoder subdirs ---
    encoder_cfg = prunable_types.get("encoders", {})
    encoder_base = _PROJECT_ROOT / encoder_cfg.get("base_path", "logic/src/models/subnets/encoders")
    always_keep_enc: Set[str] = set(encoder_cfg.get("always_keep", ["common"]))
    all_enc_subdirs: List[str] = encoder_cfg.get("all_subdirs", [])
    _prune_subdirs(encoder_base, all_enc_subdirs, needed_encoders | always_keep_enc, "encoder", dry_run)

    # --- Prune factory files ---
    factory_cfg = prunable_types.get("factories", {})
    factory_base = _PROJECT_ROOT / factory_cfg.get("base_path", "logic/src/models/subnets/factories")
    always_keep_fac: Set[str] = set(factory_cfg.get("always_keep", ["attention.py", "base.py", "__init__.py"]))
    all_fac_files: List[str] = factory_cfg.get("all_files", [])
    _prune_files(factory_base, all_fac_files, needed_factories | always_keep_fac, "factory", dry_run)


def _prune_subdirs(
    base: Path,
    all_subdirs: List[str],
    keep: Set[str],
    label: str,
    dry_run: bool,
) -> None:
    import shutil  # noqa: PLC0415
    for subdir in all_subdirs:
        if subdir in keep:
            continue
        target = base / subdir
        if not target.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove {label} subdir: {target.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing {label} subdir: {target.relative_to(_PROJECT_ROOT)}")
            shutil.rmtree(target)


def _prune_files(
    base: Path,
    all_files: List[str],
    keep: Set[str],
    label: str,
    dry_run: bool,
) -> None:
    for fname in all_files:
        if fname in keep:
            continue
        target = base / fname
        if not target.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove {label} file: {target.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing {label} file: {target.relative_to(_PROJECT_ROOT)}")
            target.unlink()


def _remove_constants_init_import(module_name: str) -> None:
    """Comment out a wildcard import from logic/src/constants/__init__.py."""
    import re  # noqa: PLC0415

    init_path = _PROJECT_ROOT / "logic/src/constants/__init__.py"
    if not init_path.exists():
        return
    content = init_path.read_text(encoding="utf-8")
    pattern = rf"(?m)^(from logic\.src\.constants\.{re.escape(module_name)} import \*.*)$"
    new_content = re.sub(pattern, r"# \1  # AUTO-REMOVED", content)
    if new_content != content:
        _log(f"  Updated constants/__init__.py: commented out {module_name} import")
        init_path.write_text(new_content, encoding="utf-8")


def remove_logic_dev_dirs(dry_run: bool) -> None:
    """Remove dev-only logic/ subdirectories and files not needed at runtime."""
    import shutil  # noqa: PLC0415

    targets = [
        _PROJECT_ROOT / "logic" / "benchmark",
        _PROJECT_ROOT / "logic" / "docs",
        _PROJECT_ROOT / "logic" / "examples",
        _PROJECT_ROOT / "logic" / "test",
        _PROJECT_ROOT / "logic" / "src" / "py.typed",
        # testing.py is only used by logic/test/ (removed above); plotting.py
        # is only used by the expo/output utilities (always removed from shared_always).
        # stats.py is removed by remove_cli.py since it is CLI-only.
        _PROJECT_ROOT / "logic" / "src" / "constants" / "testing.py",
        _PROJECT_ROOT / "logic" / "src" / "constants" / "plotting.py",
    ]
    removed_constant_modules: List[str] = []
    for path in targets:
        if not path.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove: {path.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing: {path.relative_to(_PROJECT_ROOT)}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
                if path.parent.name == "constants" and path.suffix == ".py":
                    removed_constant_modules.append(path.stem)

    if not dry_run:
        for mod in removed_constant_modules:
            _remove_constants_init_import(mod)


def remove_packages_self(dry_run: bool) -> None:
    """Self-destruct: remove logic/src/utils/packages/ as the very last pruning step."""
    import shutil  # noqa: PLC0415

    packages_dir = _PACKAGES_DIR
    if not packages_dir.exists():
        return
    if dry_run:
        _log(f"  [DRY-RUN] Would self-remove: {packages_dir.relative_to(_PROJECT_ROOT)}")
    else:
        _log(f"  Self-removing packages dir: {packages_dir.relative_to(_PROJECT_ROOT)}")
        shutil.rmtree(packages_dir)


def prune_category(
    category_key: str,
    keep_acronyms: Optional[List[str]],
    all_by_category: Dict[str, Dict[str, str]],
    config: Dict,
    dry_run: bool,
) -> int:
    """Prune a single algorithm category.

    Args:
        category_key: JSON category key (e.g. ``"constructor"``).
        keep_acronyms: User-facing acronyms to RETAIN (``None`` = keep all).
        all_by_category: Pre-computed mapping of all known acronyms per category.
        config: Parsed ``export_config.json`` dict.
        dry_run: If True, log what would be deleted without actually deleting.

    Returns:
        Number of algorithms pruned.
    """
    if keep_acronyms is None:
        _log(f"Category '{category_key}': no flag provided — keeping everything.")
        return 0

    known: Dict[str, str] = all_by_category.get(category_key, {})
    keep_upper: Set[str] = {a.upper() for a in keep_acronyms}

    _validate_acronyms(keep_upper, set(known.keys()), category_key)

    to_prune: List[str] = [
        acr for acr in sorted(known.keys()) if acr not in keep_upper
    ]

    if not to_prune:
        _log(f"Category '{category_key}': nothing to prune.")
        return 0

    kept_display = sorted(keep_upper & set(known.keys())) or ["(none)"]
    _log(
        f"Category '{category_key}': keeping {kept_display}, "
        f"pruning {len(to_prune)} algorithm(s)."
    )

    cleanup_kwargs = _build_cleanup_kwargs(category_key, config)
    pruned = 0

    for user_acr in to_prune:
        internal_acr = known[user_acr]
        if dry_run:
            _log(f"  [DRY-RUN] would prune {user_acr!r} (internal: {internal_acr!r})")
        else:
            _log(f"  Pruning {user_acr!r} (internal: {internal_acr!r}) …")
            clean_by_acronym(acronym=internal_acr, **cleanup_kwargs)
        pruned += 1

    return pruned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prune_codebase.py",
        description=(
            "Prune unselected algorithms from the WSmart-Route codebase. "
            "Each --<category> flag accepts a space-separated list of "
            "user-facing acronyms to KEEP. Omitting a flag keeps the entire "
            "category intact. Passing the flag with no values prunes the whole "
            "category."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--constructors",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Route construction algorithms to keep (e.g. HGS ALNS BPC).",
    )
    p.add_argument(
        "--selectors",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Mandatory selection policies to keep (e.g. MS_REGULAR MS_PPK).",
    )
    p.add_argument(
        "--improvement",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Route improvement operators to keep (e.g. RI_OROPT RI_S2OPT).",
    )
    p.add_argument(
        "--acceptance",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Acceptance criteria to keep (e.g. AC_GD AC_OI AC_SA).",
    )
    p.add_argument(
        "--joint",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Joint selection-and-construction methods to keep (e.g. JGO JSA).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Neural model architectures to keep (e.g. AM TAM MDAM).",
    )
    p.add_argument(
        "--rl-algorithms",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        dest="rl_algorithms",
        help="RL training algorithms to keep (e.g. RL_REINFORCE RL_PPO).",
    )
    p.add_argument(
        "--imitation-policies",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        dest="imitation_policies",
        help="Imitation learning policy types to keep (e.g. IP_HGS IP_ALNS).",
    )
    p.add_argument(
        "--drop-features",
        nargs="*",
        metavar="FEATURE",
        default=None,
        dest="drop_features",
        help=(
            "Optional pipeline features to REMOVE (e.g. META_LEARNING HPO SECURITY). "
            "Available: META_LEARNING, HPO, EVAL, SECURITY, CALLBACKS, ENUMS, UI_LOGIC, DATA_WEB. "
            "Omitting this flag keeps all features."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be deleted without actually deleting anything.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        metavar="PATH",
        help=f"Path to export_config.json (default: {_CONFIG_PATH}).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path: Path = args.config
    if not config_path.exists():
        _die(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)

    all_by_category = _collect_algorithms_by_category(config)

    if args.dry_run:
        _log("=== DRY RUN — no files will be deleted ===")

    # Map each category key → the user-provided keep list (None if flag omitted)
    category_keeps: Dict[str, Optional[List[str]]] = {
        "constructor": args.constructors,
        "selector": args.selectors,
        "improvement": args.improvement,
        "acceptance": args.acceptance,
        "joint": args.joint,
        "model": args.models,
        "rl_algorithm": args.rl_algorithms,
        "imitation_policy": args.imitation_policies,
    }

    total_pruned = 0
    for cat_key, keep_list in category_keeps.items():
        total_pruned += prune_category(
            category_key=cat_key,
            keep_acronyms=keep_list,
            all_by_category=all_by_category,
            config=config,
            dry_run=args.dry_run,
        )

    # Prune model-specific subnets after the model category is decided.
    _log("Pruning model subnets …")
    prune_subnets(
        keep_model_acronyms=category_keeps.get("model"),
        config=config,
        dry_run=args.dry_run,
    )

    # Drop optional pipeline features on request.
    drop_features: List[str] = [f.upper() for f in (args.drop_features or [])]
    if drop_features:
        _log(f"Dropping optional features: {drop_features}")
        prune_optional_features(drop_features, config, dry_run=args.dry_run)
    else:
        _log("No optional features dropped (use --drop-features to remove META_LEARNING, HPO, etc.).")

    # Remove dev-only logic/ subdirs (benchmark, docs, examples, test, py.typed).
    _log("Removing dev-only logic/ subdirectories …")
    remove_logic_dev_dirs(dry_run=args.dry_run)

    # Self-destruct: remove this packages dir last so all remove_*.py scripts are
    # available for the entire pruning run.
    _log("Removing logic/src/utils/packages/ (self-cleanup) …")
    remove_packages_self(dry_run=args.dry_run)

    _log(
        f"{'[DRY-RUN] ' if args.dry_run else ''}Done. "
        f"Total algorithms pruned: {total_pruned}."
    )


if __name__ == "__main__":
    main()
