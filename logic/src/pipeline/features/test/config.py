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

            # If custom_overrides is already a fully-resolved variant config (contains
            # top-level policy-level keys such as engine/mandatory_selection/route_improvement),
            # it IS the specific variant — store it directly rather than loading the full YAML
            # and merging.  The full-YAML merge path leaves sibling variants (e.g. og_a/og_b)
            # inside the stored config, which causes find_policy_keys to discover the wrong
            # mandatory_selection when it recurses over all sub-trees.
            _override_keys: set = set(custom_overrides.keys()) if isinstance(custom_overrides, dict) else set()
            _is_resolved_variant = bool(
                _override_keys & {"engine", "mandatory_selection", "route_improvement", "time_limit", "params"}
            )
            if custom_overrides and _is_resolved_variant:
                # Use the resolved variant config directly, wrapped under the policy name key
                final_cfg = {pol_name: copy.deepcopy(dict(custom_overrides))}
            else:
                final_cfg = copy.deepcopy(custom_cfg or cfg_path)
                if isinstance(final_cfg, str):
                    # Always load the config if it's just a path
                    final_cfg = load_config(final_cfg)

                if custom_overrides and final_cfg and isinstance(final_cfg, dict):
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

    Checks ``WSR_POLICY_CONFIG_DIR`` (set by the batch parallel scheduler for
    per-job YAML isolation) before falling back to the canonical policies dir.

    Args:
        pol_name: Policy name.

    Returns:
        Configuration file path.
    """
    canonical = os.path.join(udef.ROOT_DIR, "logic", "configs", "policies")
    custom = os.environ.get("WSR_POLICY_CONFIG_DIR", "")
    # Search custom dir first (contains job-specific patched copies), then canonical.
    search_dirs = [d for d in [custom, canonical] if d]

    def _check_dirs(filename: str) -> str:
        for d in search_dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                return p
        return ""

    # Try direct mapping first
    for fname in (f"{pol_name}.yaml", f"policy_{pol_name}.yaml" if not pol_name.startswith("policy_") else ""):
        if fname:
            found = _check_dirs(fname)
            if found:
                return found

    # Try prefix-based matching
    parts = pol_name.split("_")
    for i in range(len(parts), 0, -1):
        prefix = "_".join(parts[:i])
        for fname in (f"{prefix}.yaml", f"policy_{prefix}.yaml"):
            found = _check_dirs(fname)
            if found:
                return found

    # Fall back to individual parts
    for part in parts:
        for fname in (f"{part}.yaml", f"policy_{part}.yaml"):
            found = _check_dirs(fname)
            if found:
                return found

    return ""


def _extract_variants(pol_name: str, cfg_path: str) -> Tuple[List[Tuple[str, str, Any]], Any]:  # noqa: C901
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

        inner_cfg, variant_name = _find_inner_config(pol_cfg, pol_name)
        ms_list, ac_list, pp_list, ms_idx, ac_idx = _parse_inner_components(inner_cfg)

        ms_list = _expand_dict_ms_list(ms_list)
        ac_list = _expand_dict_ms_list(ac_list)
        pp_list = _expand_dict_ms_list(pp_list)
        ms_list_iter = ms_list if ms_list else [None]
        ac_list_iter = ac_list if ac_list else [None]

        # Flatten the inner_cfg into a single dictionary
        merged_cfg = {}
        for item in inner_cfg:
            if isinstance(item, dict) or hasattr(item, "items"):
                merged_cfg.update(dict(item))

        # If neither has multiple items, treat as a single variant
        if len(ms_list_iter) <= 1 and len(ac_list_iter) <= 1:
            var_cfg = copy.deepcopy(merged_cfg)
            if ms_list_iter and ms_list_iter[0]:
                var_cfg["mandatory_selection"] = [ms_list_iter[0]]
            if ac_list_iter and ac_list_iter[0]:
                var_cfg["acceptance_criteria"] = [ac_list_iter[0]]

            prefix = f"{_clean_id(ms_list[0], 'ms_')}_" if ms_list else ""
            ac_suffix = f"_{_clean_id(ac_list[0], 'ac_')}" if ac_list else ""
            pp_suffix = f"_{_clean_id(pp_list[0], 'ri_')}" if pp_list else ""
            suffix = f"{ac_suffix}{pp_suffix}"
            return [(prefix, suffix, var_cfg)], variant_name

        variants: List[Tuple[str, str, Any]] = []
        for ms_item in ms_list_iter:
            for ac_item in ac_list_iter:
                prefix = f"{_clean_id(ms_item, 'ms_')}_" if ms_item else ""
                ac_suffix = f"_{_clean_id(ac_item, 'ac_')}" if ac_item else ""
                pp_suffix = f"_{_clean_id(pp_list[0], 'ri_')}" if pp_list else ""
                suffix = f"{ac_suffix}{pp_suffix}"

                # If the policy name already contains the prefix, we are likely looking at an expanded name
                # In that case, only yield the variant that matches this prefix
                clean_prefix = prefix.rstrip("_")
                if clean_prefix and clean_prefix in pol_name:
                    var_cfg = copy.deepcopy(merged_cfg)
                    if ms_item:
                        var_cfg["mandatory_selection"] = [ms_item]
                    if ac_item:
                        var_cfg["acceptance_criteria"] = [ac_item]
                    return [(prefix, suffix, var_cfg)], variant_name

                var_cfg = copy.deepcopy(merged_cfg)
                if ms_item:
                    var_cfg["mandatory_selection"] = [ms_item]
                if ac_item:
                    var_cfg["acceptance_criteria"] = [ac_item]
                variants.append((prefix, suffix, var_cfg))
        return variants, variant_name

    except Exception as e:
        print(f"Warning: Could not load variants for {pol_name}: {e}")
        return [("", "", None)], None


def _find_inner_config(pol_cfg: Any, pol_name: str = "") -> Tuple[Any, Any]:
    """Find the list of configurations/variants within a policy config.

    Args:
        pol_cfg: Policy configuration object.
        pol_name: Policy name to use for matching the exact variant.

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
                if any(
                    k in v_dict
                    for k in [
                        "mandatory",
                        "mandatory_selection",
                        "route_improvement",
                        "acceptance_criteria",
                        "engine",
                        "params",
                    ]
                ):
                    return [v], _k
                # Otherwise, look one level deeper
                valid_subs = []
                for sub_k, sub_v in v_dict.items():
                    if isinstance(sub_v, list):
                        valid_subs.append((sub_v, sub_k))
                        continue
                    sub_v_obj: object = sub_v
                    if isinstance(sub_v_obj, ITraversable) or hasattr(sub_v_obj, "items"):
                        sub_v_dict = cast(Dict[str, Any], sub_v_obj)
                        if any(
                            sk in sub_v_dict
                            for sk in [
                                "mandatory",
                                "mandatory_selection",
                                "route_improvement",
                                "acceptance_criteria",
                                "engine",
                                "params",
                            ]
                        ):
                            valid_subs.append(([sub_v], sub_k))

                if valid_subs:
                    if pol_name:
                        for sub_v, sub_k in valid_subs:
                            if sub_k and sub_k in pol_name:
                                return sub_v, sub_k
                    return valid_subs[0][0], valid_subs[0][1]
    return [], None


def _parse_inner_components(
    inner_cfg: Any,
) -> Tuple[List[Any], List[Any], List[Any], int, int]:
    """Extract mandatory, acceptance criteria, and route improvement lists.

    Args:
        inner_cfg: Inner configuration object.

    Returns:
        Tuple of (ms_list, ac_list, pp_list, ms_idx, ac_idx).
    """
    ms_list: List[Any] = []
    ac_list: List[Any] = []
    pp_list: List[Any] = []
    ms_idx = -1
    ac_idx = -1

    inner_cfg_list = [inner_cfg] if isinstance(inner_cfg, dict) else inner_cfg
    for idx, item in enumerate(inner_cfg_list):
        item_obj: object = item
        if isinstance(item_obj, ITraversable) or hasattr(item_obj, "items"):
            item_dict = dict(item_obj)
            if "mandatory_selection" in item_dict:
                ms_list = item_dict["mandatory_selection"]
                ms_idx = idx
            elif "mandatory" in item_dict:
                ms_list = item_dict["mandatory"]
                ms_idx = idx

            if "acceptance_criteria" in item_dict:
                ac_list = item_dict["acceptance_criteria"]
                ac_idx = idx

            if "route_improvement" in item_dict:
                pp_list = item_dict["route_improvement"]

    return ms_list, ac_list, pp_list, ms_idx, ac_idx


def _apply_overrides(var_cfg: Any, ms_idx: int, ms_item: Any, ac_idx: int, ac_item: Any) -> None:
    """Apply mandatory and acceptance criteria overrides to a deep-copied config.

    Args:
        var_cfg: Deep-copied policy configuration object.
        ms_idx: Index of the mandatory config to modify.
        ms_item: Mandatory list item to apply.
        ac_idx: Index of the acceptance criteria config to modify.
        ac_item: Acceptance criteria list item to apply.
    """
    var_inner, _ = _find_inner_config(var_cfg)
    if var_inner:
        if isinstance(var_inner, list):
            if ms_item is not None and 0 <= ms_idx < len(var_inner):
                item_obj = var_inner[ms_idx]
                if isinstance(item_obj, (dict, ITraversable)):
                    if "mandatory_selection" in item_obj:
                        cast(Any, item_obj)["mandatory_selection"] = [ms_item]
                    else:
                        cast(Any, item_obj)["mandatory"] = [ms_item]
            if ac_item is not None and 0 <= ac_idx < len(var_inner):
                item_obj = var_inner[ac_idx]
                if isinstance(item_obj, (dict, ITraversable)):
                    cast(Any, item_obj)["acceptance_criteria"] = [ac_item]
        else:  # dict
            if ms_item is not None:
                if "mandatory_selection" in var_inner:
                    var_inner["mandatory_selection"] = [ms_item]
                elif "mandatory" in var_inner:
                    var_inner["mandatory"] = [ms_item]
            if ac_item is not None:
                var_inner["acceptance_criteria"] = [ac_item]


def _expand_dict_ms_list(ms_list: Any) -> List[Any]:
    """Expand a dict-format mandatory_selection into a flat list of single-variant dicts.

    A dict like ``{"other/ms_last_minute.yaml": ["cf70", "cf90"]}`` expands to
    ``[{"other/ms_last_minute.yaml": "cf70"}, {"other/ms_last_minute.yaml": "cf90"}]``.
    Plain lists of strings or already-flat lists are returned unchanged.

    Args:
        ms_list: Raw value from the mandatory_selection key.

    Returns:
        Flat list of items (strings or single-variant dicts ``{file: variant}``).
    """
    _is_mapping = isinstance(ms_list, dict) or (hasattr(ms_list, "items") and not isinstance(ms_list, (str, list)))
    if _is_mapping and len(ms_list) == 1:
        file_path, variants = next(iter(ms_list.items()))
        if isinstance(variants, (list, tuple)) or (hasattr(variants, "__iter__") and not isinstance(variants, str)):
            return [{str(file_path): str(v)} for v in variants]
        return [{str(file_path): str(variants)}]
    if isinstance(ms_list, (list, tuple)) or (
        hasattr(ms_list, "__iter__") and not isinstance(ms_list, (str, dict)) and not hasattr(ms_list, "items")
    ):
        expanded: List[Any] = []
        for item in ms_list:
            _item_is_mapping = isinstance(item, dict) or (hasattr(item, "items") and not isinstance(item, str))
            if _item_is_mapping and len(item) == 1:
                file_path, variants = next(iter(item.items()))
                if isinstance(variants, (list, tuple)) or (
                    hasattr(variants, "__iter__") and not isinstance(variants, str)
                ):
                    expanded.extend({str(file_path): str(v)} for v in variants)
                else:
                    expanded.append({str(file_path): str(variants)})
            else:
                expanded.append(item)
        return expanded
    return ms_list if ms_list else []


def _clean_id(path_or_str: Any, prefix: str) -> str:
    """Clean a component ID from a path or string.

    Args:
        path_or_str: Path or string to clean.
        prefix: Prefix to remove.

    Returns:
        Cleaned identifier.
    """
    if isinstance(path_or_str, dict) and len(path_or_str) == 1:
        file_path, variant_key = next(iter(path_or_str.items()))
        variant_key = str(variant_key) if variant_key is not None else ""
        if variant_key and variant_key != "default":
            return variant_key
        name = os.path.basename(str(file_path))
        for p in [prefix, ".xml", ".yaml"]:
            name = name.replace(p, "")
        return name
    if not isinstance(path_or_str, str):
        return ""
    name = os.path.basename(path_or_str)
    for p in [prefix, ".xml", ".yaml"]:
        name = name.replace(p, "")
    return name
