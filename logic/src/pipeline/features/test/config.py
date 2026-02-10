"""
Configuration loader for Test Engine.
"""

import copy
import os

import logic.src.constants as udef
from logic.src.utils.configs.config_loader import load_config
from logic.src.interfaces import ITraversable


def expand_policy_configs(opts):
    """
    Expands policy names into full configuration paths and variants.
    Modifies opts["policies"] and opts["config_path"] in place.
    """
    policies = []

    for pol in opts["policies"]:
        tmp_pols = [pol]
        for tmp_pol in tmp_pols:
            prefix_str = ""
            suffix_str = ""
            variant_name = None
            cfg_path = None
            variants = [("", "", None)]

            try:
                cfg_path = os.path.join(udef.ROOT_DIR, "assets", "configs", "policies", f"{tmp_pol}.yaml")
                if not os.path.exists(cfg_path) and not tmp_pol.startswith("policy_"):
                    cfg_path = os.path.join(udef.ROOT_DIR, "assets", "configs", "policies", f"policy_{tmp_pol}.yaml")

                if os.path.exists(cfg_path):
                    pol_cfg = load_config(cfg_path)
                    inner_cfg = []
                    if pol_cfg:
                        for _k, v in pol_cfg.items():
                            if isinstance(v, list):
                                inner_cfg = v
                                break
                            if isinstance(v, ITraversable):
                                if "must_go" in v or "post_processing" in v:
                                    pass
                                else:
                                    for sub_k, sub_v in v.items():
                                        if isinstance(sub_v, list):
                                            inner_cfg = sub_v
                                            variant_name = sub_k
                                            break

                    mg_list = []
                    pp_list = []
                    match_item_idx = -1
                    for idx, item in enumerate(inner_cfg):
                        if isinstance(item, ITraversable):
                            if "must_go" in item:
                                mg_list = item["must_go"]
                                match_item_idx = idx
                            if "post_processing" in item:
                                pp_list = item["post_processing"]

                    variants = []
                    if mg_list and len(mg_list) > 1:
                        for mg_item in mg_list:
                            v_prefix = ""
                            if isinstance(mg_item, str):
                                clean_mg = (
                                    os.path.basename(mg_item)
                                    .replace("mg_", "")
                                    .replace(".xml", "")
                                    .replace(".yaml", "")
                                )
                                v_prefix = f"{clean_mg}_"

                            v_suffix = ""
                            if pp_list:
                                first_pp = pp_list[0]
                                if isinstance(first_pp, str):
                                    clean_pp = (
                                        os.path.basename(first_pp)
                                        .replace("pp_", "")
                                        .replace(".xml", "")
                                        .replace(".yaml", "")
                                    )
                                    v_suffix = f"_{clean_pp}"

                            var_cfg = copy.deepcopy(pol_cfg)
                            var_inner = []
                            if var_cfg:
                                found = False
                                for _k, v in var_cfg.items():
                                    if isinstance(v, list):
                                        var_inner = v
                                        found = True
                                        break
                                    if isinstance(v, ITraversable):
                                        if "must_go" in v or "post_processing" in v:
                                            pass
                                        else:
                                            for sub_k, sub_v in v.items():
                                                if isinstance(sub_v, list):
                                                    var_inner = sub_v
                                                    found = True
                                                    break
                                    if found:
                                        break

                            if var_inner and match_item_idx >= 0 and match_item_idx < len(var_inner):
                                if isinstance(var_inner[match_item_idx], ITraversable):
                                    var_inner[match_item_idx]["must_go"] = [mg_item]

                            variants.append((v_prefix, v_suffix, var_cfg))
                    else:
                        prefix_str = ""
                        if mg_list:
                            first_mg = mg_list[0]
                            if isinstance(first_mg, str):
                                clean_mg = (
                                    os.path.basename(first_mg)
                                    .replace("mg_", "")
                                    .replace(".xml", "")
                                    .replace(".yaml", "")
                                )
                                prefix_str = f"{clean_mg}_"

                        suffix_str = ""
                        if pp_list:
                            first_pp = pp_list[0]
                            if isinstance(first_pp, str):
                                clean_pp = (
                                    os.path.basename(first_pp)
                                    .replace("pp_", "")
                                    .replace(".xml", "")
                                    .replace(".yaml", "")
                                )
                                suffix_str = f"_{clean_pp}"

                        variants.append((prefix_str, suffix_str, None))

            except Exception as e:
                print(f"Warning: Could not load config for naming {tmp_pol}: {e}")
                variants = [("", "", None)]

            for prefix, suffix, custom_cfg in variants:
                middle_name = tmp_pol.replace("policy_", "")
                if variant_name and variant_name.lower() != "default":
                    middle_name = f"{middle_name}_{variant_name}"

                full_name = f"{prefix}{middle_name}{suffix}_{opts['data_distribution']}"
                policies.append(full_name)

                if "config_path" not in opts or not isinstance(opts["config_path"], ITraversable):
                    opts["config_path"] = (
                        opts.get("config_path", {}) if isinstance(opts.get("config_path"), ITraversable) else {}
                    )

                key = full_name
                if custom_cfg:
                    opts["config_path"][key] = custom_cfg
                elif cfg_path and os.path.exists(cfg_path):
                    opts["config_path"][key] = cfg_path

    opts["policies"] = policies
