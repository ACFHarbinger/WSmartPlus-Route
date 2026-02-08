"""
Environment and cost setup utilities.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import gurobipy as gp
from dotenv import dotenv_values

import logic.src.constants as udef
from logic.src.utils.security import decrypt_file_data, load_key


def setup_cost_weights(opts: Dict[str, Any], def_val: float = 1.0) -> Dict[str, float]:
    """
    Sets up the cost weights dictionary based on problem type.

    Args:
        opts: Options dictionary.
        def_val: Default weight value. Defaults to 1.0.

    Returns:
        Dictionary of cost weights (waste, length, overflows, etc.).
    """

    def _set_val(cost_weight: Optional[float], default_value: float) -> float:
        return default_value if cost_weight is None else cost_weight

    cw_dict: Dict[str, float] = {}
    if opts["problem"] in udef.PROBLEMS:  # type: ignore
        cw_dict["waste"] = opts["w_waste"] = _set_val(opts["w_waste"], def_val)
        cw_dict["length"] = opts["w_length"] = _set_val(opts["w_length"], def_val)
        if "overflows" in opts or opts["problem"] in [
            "wcvrp",
            "cwcvrp",
            "sdwcvrp",
            "scwcvrp",
        ]:
            cw_dict["overflows"] = opts["w_overflows"] = _set_val(opts.get("w_overflows"), def_val)
    return cw_dict


def setup_env(
    policy: str,
    server: bool = False,
    gplic_filename: Optional[str] = None,
    symkey_name: Optional[str] = None,
    env_filename: Optional[str] = None,
) -> Optional[gp.Env]:
    """
    Sets up the solver environment (e.g., Gurobi).

    Args:
        policy: Policy name to determine environment type.
        server: Whether running on a server (requires specific license handling).
        gplic_filename: Gurobi license filename.
        symkey_name: Symmetric key name for decryption.
        env_filename: Environment variables filename.

    Returns:
        The Gurobi environment, or None if not applicable.
    """
    if "vrpp" in policy and "hexaly" not in policy:
        params: Dict[str, Any] = {}
        if server:

            def convert_int(param: str) -> Union[int, str]:
                """Helper to convert string parameters to int if possible."""
                return int(param) if param.isdigit() else param

            if gplic_filename is not None:
                gplic_path: str = os.path.join(udef.ROOT_DIR, "assets", "api", gplic_filename)
                if symkey_name:
                    key = load_key(symkey_name=symkey_name, env_filename=env_filename or ".env")
                    data = decrypt_file_data(key, gplic_path)
                else:
                    with open(gplic_path, "r") as gp_file:
                        data = gp_file.read()
                params = {
                    line.split("=")[0]: convert_int(line.split("=")[1]) for line in data.split("\n") if "=" in line
                }
            else:
                assert env_filename is not None
                env_path: str = os.path.join(udef.ROOT_DIR, "env", env_filename)
                config: Dict[str, Optional[str]] = dotenv_values(env_path)
                glp_ls: List[str] = ["WLSACCESSID", "WLSSECRET", "LICENSEID"]
                params = {glp: convert_int(config.get(glp, "")) for glp in glp_ls}  # type: ignore
                for glp_key, glp_val in params.items():
                    if isinstance(glp_val, str) and glp_val == "":
                        raise ValueError(f"Missing parameter {glp_key} for Gurobi license")
        else:
            if gplic_filename is not None:
                gplic_path = os.path.join(udef.ROOT_DIR, "assets", "api", gplic_filename)
                if os.path.exists(gplic_path):
                    os.environ["GRB_LICENSE_FILE"] = gplic_path
        params["OutputFlag"] = udef.SOLVER_OUTPUT_FLAG
        return gp.Env(params=params)
    return None
