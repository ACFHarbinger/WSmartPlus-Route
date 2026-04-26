"""
Configuration loader for Test Engine.

Attributes:
    expand_policy_configs: Expands policy names into full configuration paths and variants.
    _resolve_policy_cfg_path: Resolve the YAML configuration path for a policy.
    _extract_variants: Extract policy variants from its configuration.
    _find_inner_config: Find the list of configurations/variants within a policy config.
    _parse_inner_components: Parse the inner components (MS and PP) from a policy configuration.
    _apply_ms_override: Apply the multi-start (MS) override to a policy configuration.
    _clean_id: Clean an identifier string by removing a prefix and replacing underscores with hyphens.

Example:
    >>> from logic.src.pipeline.features.test.config import expand_policy_configs
    >>> config = Config()
    >>> expand_policy_configs(config)
"""

import copy
import os
from typing import Any, Dict, List, Tuple, cast

from omegaconf import OmegaConf

import logic.src.constants as udef
from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.utils.configs.config_loader import load_config


def expand_policy_configs(cfg: Config) -> None:  # noqa: C901
    """
    Expands policy names into full configuration paths and variants.

    Populates ``cfg.sim.full_policies`` with expanded policy names and
    ``cfg.sim.config_path`` with their loaded configuration dicts.

    Args:
        cfg: Root configuration object. Reads from ``cfg.sim.policies``
            and ``cfg.sim.data_distribution``.
    """
    sim = cfg.sim
    policies: List[str] = []

    config_path: Dict[str, Any] = dict(sim.config_path) if sim.config_path else {}
    dist_suffix = f"_{sim.data_distribution}"
    resolved_policies = OmegaConf.to_container(sim.policies, resolve=True)
    if not isinstance(resolved_policies, list):
        resolved_policies = [resolved_policies]

    for item in resolved_policies:
        if isinstance(item, dict) and len(item) == 1:
            pol_name = list(item.keys())[0]
            custom_overrides = item[pol_name]
        else:
            pol_name = str(item)
            custom_overrides = None

        cfg_path = _resolve_policy_cfg_path(pol_name)
        variants, variant_name = _extract_variants(pol_name, cfg_path)

        for prefix, suffix, custom_cfg in variants:
            middle_name = pol_name.replace("policy_", "")
            # Only append variant name if it's not already in the middle_name
            if variant_name and variant_name.lower() != "default" and variant_name.lower() not in middle_name.lower():
                middle_name = f"{middle_name}_{variant_name}"

            # Prevent doubling of prefixes if the policy name is already expanded
            full_name = pol_name if prefix and prefix in pol_name else f"{prefix}{middle_name}{suffix}"

            if not full_name.endswith(dist_suffix):
                full_name = f"{full_name}{dist_suffix}"

            final_cfg = copy.deepcopy(custom_cfg or cfg_path)
            if custom_overrides and final_cfg:
                if isinstance(final_cfg, str):
                    # Load it first so we can merge
                    final_cfg = load_config(final_cfg)

                if isinstance(final_cfg, dict):
                    # Special handling for single-key policy configs
                    pol_key = list(final_cfg.keys())[0] if len(final_cfg) == 1 else None
                    if pol_key and pol_key == pol_name:
                        # Ensure both are flattened if they are lists (OmegaConf list-of-dicts style)
                        target = final_cfg[pol_key]
                        if isinstance(target, (list, tuple)) or (
                            hasattr(target, "__iter__") and not hasattr(target, "update")
                        ):
                            merged_target = {}
                            for item in target:
                                if hasattr(item, "items"):
                                    merged_target.update(item)
                            final_cfg[pol_key] = merged_target

                        if isinstance(custom_overrides, (list, tuple)) or (
                            hasattr(custom_overrides, "__iter__") and not hasattr(custom_overrides, "update")
                        ):
                            merged_overrides = {}
                            for item in custom_overrides:
                                if hasattr(item, "items"):
                                    merged_overrides.update(item)
                            custom_overrides = merged_overrides

                        final_cfg[pol_key].update(custom_overrides)
                    else:
                        if hasattr(final_cfg, "update"):
                            # Handle case where custom_overrides might be a list here too
                            if isinstance(custom_overrides, (list, tuple)) or (
                                hasattr(custom_overrides, "__iter__") and not hasattr(custom_overrides, "update")
                            ):
                                merged_overrides = {}
                                for item in custom_overrides:
                                    if hasattr(item, "items"):
                                        merged_overrides.update(item)
                                custom_overrides = merged_overrides
                            final_cfg.update(custom_overrides)

            policies.append(full_name)
            config_path[full_name] = final_cfg

    sim.full_policies = policies
    sim.config_path = config_path


def _resolve_policy_cfg_path(pol_name: str) -> str:
    """Resolve the YAML configuration path for a policy.

    Args:
        pol_name: Policy name.

    Returns:
        Configuration file path.
    """
    base_dir = os.path.join(udef.ROOT_DIR, "logic", "configs", "policies")

    # Try direct mapping first
    paths_to_check = [
        os.path.join(base_dir, f"{pol_name}.yaml"),
        os.path.join(base_dir, f"policy_{pol_name}.yaml") if not pol_name.startswith("policy_") else None,
    ]
    for p in paths_to_check:
        if p and os.path.exists(p):
            return p

    # If not found, try to find a base policy file by splitting
    parts = pol_name.split("_")
    for i in range(len(parts), 0, -1):
        prefix = "_".join(parts[:i])
        paths = [
            os.path.join(base_dir, f"{prefix}.yaml"),
            os.path.join(base_dir, f"policy_{prefix}.yaml"),
        ]
        for p in paths:
            if os.path.exists(p):
                return p

    # If still not found, check individual parts (e.g. regular_lvl3_cvrp_ortools -> cvrp)
    for part in parts:
        paths = [
            os.path.join(base_dir, f"{part}.yaml"),
            os.path.join(base_dir, f"policy_{part}.yaml"),
        ]
        for p in paths:
            if os.path.exists(p):
                return p

    return ""


def _extract_variants(pol_name: str, cfg_path: str) -> Tuple[List[Tuple[str, str, Any]], Any]:
    """Extract policy variants from its configuration.

    Args:
        pol_name: Policy name.
        cfg_path: Configuration file path.

    Returns:
        Tuple of (variants, variant_name).
    """
    if not cfg_path:
        return [("", "", None)], None

    try:
        pol_cfg = load_config(cfg_path)
        if not pol_cfg:
            return [("", "", None)], None

        inner_cfg, variant_name = _find_inner_config(pol_cfg)
        ms_list, pp_list, match_idx = _parse_inner_components(inner_cfg)

        if ms_list and len(ms_list) > 1:
            variants: List[Tuple[str, str, Any]] = []
            for ms_item in ms_list:
                prefix = f"{_clean_id(ms_item, 'ms_')}_"
                suffix = f"_{_clean_id(pp_list[0], 'ri_')}" if pp_list else ""

                # If the policy name already contains the prefix, we are likely looking at an expanded name
                # In that case, only yield the variant that matches this prefix
                clean_prefix = prefix.rstrip("_")
                if clean_prefix in pol_name:
                    var_cfg = copy.deepcopy(pol_cfg)
                    _apply_ms_override(var_cfg, match_idx, ms_item)
                    return [(prefix, suffix, var_cfg)], variant_name

                var_cfg = copy.deepcopy(pol_cfg)
                _apply_ms_override(var_cfg, match_idx, ms_item)
                variants.append((prefix, suffix, var_cfg))
            return variants, variant_name

        # Single variant case
        prefix = f"{_clean_id(ms_list[0], 'ms_')}_" if ms_list else ""
        suffix = f"_{_clean_id(pp_list[0], 'ri_')}" if pp_list else ""
        return [(prefix, suffix, None)], variant_name

    except Exception as e:
        print(f"Warning: Could not load variants for {pol_name}: {e}")
        return [("", "", None)], None


def _find_inner_config(pol_cfg: Any) -> Tuple[Any, Any]:
    """Find the list of configurations/variants within a policy config.

    Args:
        pol_cfg: Policy configuration object.

    Returns:
        Tuple of (inner_config, variant_name).
    """
    pol_cfg_obj: object = pol_cfg
    if isinstance(pol_cfg_obj, ITraversable) or hasattr(pol_cfg_obj, "items"):
        pol_cfg_dict = cast(Dict[str, Any], pol_cfg_obj)
        for _k, v in pol_cfg_dict.items():
            if isinstance(v, list):
                return v, None
            v_obj: object = v
            if isinstance(v_obj, ITraversable) or hasattr(v_obj, "items"):
                v_dict = cast(Dict[str, Any], v_obj)
                # If this dict itself contains components, it's an inner variant
                if any(k in v_dict for k in ["mandatory", "route_improvement", "engine", "params"]):
                    return [v], _k
                # Otherwise, look one level deeper
                for sub_k, sub_v in v_dict.items():
                    if isinstance(sub_v, list):
                        return sub_v, sub_k
                    sub_v_obj: object = sub_v
                    if isinstance(sub_v_obj, ITraversable) or hasattr(sub_v_obj, "items"):
                        sub_v_dict = cast(Dict[str, Any], sub_v_obj)
                        if any(sk in sub_v_dict for sk in ["mandatory", "route_improvement", "engine", "params"]):
                            return [sub_v], sub_k
    return [], None


def _parse_inner_components(
    inner_cfg: Any,
) -> Tuple[List[Any], List[Any], int]:
    """Extract mandatory and route improvement lists from inner config.

    Args:
        inner_cfg: Inner configuration object.

    Returns:
        Tuple of (mandatory_list, route_improvement_list, match_index).
    """
    ms_list: List[Any] = []
    pp_list: List[Any] = []
    match_idx = -1
    for idx, item in enumerate(inner_cfg):
        item_obj: object = item
        if isinstance(item_obj, ITraversable):
            if "mandatory" in item_obj:
                ms_list = item_obj["mandatory"]
                match_idx = idx
            if "route_improvement" in item_obj:
                pp_list = item_obj["route_improvement"]
    return ms_list, pp_list, match_idx


def _apply_ms_override(var_cfg: Any, match_idx: int, ms_item: str) -> None:
    """Apply a mandatory override to a deep-copied config.

    Args:
        var_cfg: Deep-copied policy configuration object.
        match_idx: Index of the configuration to modify.
        ms_item: Mandatory list item to apply.
    """
    var_inner, _ = _find_inner_config(var_cfg)
    if var_inner and 0 <= match_idx < len(var_inner):
        item = var_inner[match_idx]
        item_obj: object = item
        if isinstance(item_obj, (dict, ITraversable)):
            cast(Any, item_obj)["mandatory"] = [ms_item]


def _clean_id(path_or_str: Any, prefix: str) -> str:
    """Clean a component ID from a path or string.

    Args:
        path_or_str: Path or string to clean.
        prefix: Prefix to remove.

    Returns:
        Cleaned identifier.
    """
    if not isinstance(path_or_str, str):
        return ""
    name = os.path.basename(path_or_str)
    for p in [prefix, ".xml", ".yaml"]:
        name = name.replace(p, "")
    return name
