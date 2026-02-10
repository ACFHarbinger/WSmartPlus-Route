"""
Configuration loader for Test Engine.
"""

import copy
import os
from typing import Any, Dict, cast

import logic.src.constants as udef
from logic.src.interfaces import ITraversable
from logic.src.utils.configs.config_loader import load_config


def expand_policy_configs(opts):
    """
    Expands policy names into full configuration paths and variants.
    Modifies opts["policies"] and opts["config_path"] in place.
    """
    policies = []

    if "config_path" not in opts or not isinstance(opts["config_path"], (dict, ITraversable)):
        opts["config_path"] = cast(Dict[str, Any], {})

    config_path = cast(Dict[str, Any], opts["config_path"])

    for pol_name in opts["policies"]:
        cfg_path = _resolve_policy_cfg_path(pol_name)
        variants, variant_name = _extract_variants(pol_name, cfg_path)

        for prefix, suffix, custom_cfg in variants:
            middle_name = pol_name.replace("policy_", "")
            if variant_name and variant_name.lower() != "default":
                middle_name = f"{middle_name}_{variant_name}"

            full_name = f"{prefix}{middle_name}{suffix}_{opts['data_distribution']}"
            policies.append(full_name)
            config_path[full_name] = custom_cfg or cfg_path

    opts["policies"] = policies


def _resolve_policy_cfg_path(pol_name: str) -> str:
    """Resolve the YAML configuration path for a policy."""
    base_dir = os.path.join(udef.ROOT_DIR, "assets", "configs", "policies")
    paths = [
        os.path.join(base_dir, f"{pol_name}.yaml"),
        os.path.join(base_dir, f"policy_{pol_name}.yaml") if not pol_name.startswith("policy_") else None,
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


def _extract_variants(pol_name: str, cfg_path: str):
    """Extract policy variants from its configuration."""
    if not cfg_path:
        return [("", "", None)], None

    try:
        pol_cfg = load_config(cfg_path)
        if not pol_cfg:
            return [("", "", None)], None

        inner_cfg, variant_name = _find_inner_config(pol_cfg)
        mg_list, pp_list, match_idx = _parse_inner_components(inner_cfg)

        if mg_list and len(mg_list) > 1:
            variants = []
            for mg_item in mg_list:
                prefix = f"{_clean_id(mg_item, 'mg_')}_"
                suffix = f"_{_clean_id(pp_list[0], 'pp_')}" if pp_list else ""

                var_cfg = copy.deepcopy(pol_cfg)
                _apply_mg_override(var_cfg, match_idx, mg_item)
                variants.append((prefix, suffix, var_cfg))
            return variants, variant_name

        # Single variant case
        prefix = f"{_clean_id(mg_list[0], 'mg_')}_" if mg_list else ""
        suffix = f"_{_clean_id(pp_list[0], 'pp_')}" if pp_list else ""
        return [(prefix, suffix, None)], variant_name

    except Exception as e:
        print(f"Warning: Could not load variants for {pol_name}: {e}")
        return [("", "", None)], None


def _find_inner_config(pol_cfg: Any):
    """Find the list of configurations/variants within a policy config."""
    if hasattr(pol_cfg, "items"):
        for _k, v in pol_cfg.items():
            if isinstance(v, list):
                return v, None
            if isinstance(v, ITraversable):
                # Skip top-level components
                if any(k in cast(Any, v) for k in ["must_go", "post_processing"]):
                    continue
                for sub_k, sub_v in cast(Any, v).items():
                    if isinstance(sub_v, list):
                        return sub_v, sub_k
    return [], None


def _parse_inner_components(inner_cfg):
    """Extract must-go and post-processing lists from inner config."""
    mg_list, pp_list, match_idx = [], [], -1
    for idx, item in enumerate(inner_cfg):
        if isinstance(item, ITraversable):
            if "must_go" in item:
                mg_list = item["must_go"]
                match_idx = idx
            if "post_processing" in item:
                pp_list = item["post_processing"]
    return mg_list, pp_list, match_idx


def _apply_mg_override(var_cfg: Any, match_idx: int, mg_item: str):
    """Apply a must-go override to a deep-copied config."""
    var_inner, _ = _find_inner_config(var_cfg)
    if var_inner and 0 <= match_idx < len(var_inner):
        item = var_inner[match_idx]
        if isinstance(item, (dict, ITraversable)):
            cast(Any, item)["must_go"] = [mg_item]


def _clean_id(path_or_str: Any, prefix: str) -> str:
    """Clean a component ID from a path or string."""
    if not isinstance(path_or_str, str):
        return ""
    name = os.path.basename(path_or_str)
    for p in [prefix, ".xml", ".yaml"]:
        name = name.replace(p, "")
    return name
