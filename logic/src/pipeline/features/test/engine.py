"""
Simulation Test Runner.
"""

import contextlib
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

import logic.src.constants as udef
from logic.src.pipeline.simulations.repository import load_simulator_data

from .config import expand_policy_configs
from .orchestrator import simulator_testing


def run_wsr_simulator_test(opts):
    """
    Main entry point for the WSmart+ Route simulator test engine.

    This function orchestrates the end-to-end simulation testing workflow:
    1. Synchronizes Hydra configuration into a standard dictionary.
    2. Handles backwards compatibility for graph size and metadata.
    3. Initializes random seeds for reproducibility.
    4. Validates data availability and resolves data sizes.
    5. Expands policy configurations for multi-policy testing.
    6. Ensures output and checkpoint directories exist.
    7. Dispatches to the orchestrator for parallel or sequential execution.

    Args:
        opts (dict | DictConfig): Configuration options containing simulation
            parameters, policy settings, and environment metadata.
    """
    opts = _prepare_opts(opts)

    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    data_size = _resolve_data_size(opts)

    print(f"Area {opts['area']} ({data_size} available) for {opts['size']} bins")
    if data_size != opts["size"] and not opts.get("focus_graph"):
        wtype_suffix = f"_{opts['waste_type']}" if opts.get("waste_type") else ""
        opts["focus_graph"] = f"graphs_{opts['area']}_{opts['size']}V_{opts['n_samples']}N{wtype_suffix}.json"

    expand_policy_configs(opts)
    _ensure_directories(opts)

    device = torch.device("cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.device_count() - 1}")
    try:
        simulator_testing(opts, data_size, device)
    except Exception as e:
        raise Exception(f"failed to execute WSmart+ Route simulations due to {repr(e)}") from e


def _prepare_opts(opts):
    """Flatten and normalize configuration options."""
    if not isinstance(opts, dict):
        with contextlib.suppress(Exception):
            opts = OmegaConf.to_container(opts, resolve=True)

    if "num_loc" in opts:
        opts["size"] = opts["num_loc"]

    if "graph" in opts:
        g = opts["graph"]
        opts.update(
            {
                "size": g["num_loc"],
                "area": g["area"],
                "waste_type": g["waste_type"],
                "dm_filepath": g.get("dm_filepath"),
                "waste_filepath": g.get("waste_filepath"),
                "edge_threshold": g.get("edge_threshold"),
                "edge_method": g.get("edge_method"),
            }
        )
    return opts


def _resolve_data_size(opts):
    """Resolve the available data size for the given area and requested size."""
    try:
        data_tmp, _ = load_simulator_data(opts.get("data_dir"), opts["size"], opts["area"], opts["waste_type"])
        return len(data_tmp)
    except Exception:
        area, size = opts["area"], opts["size"]
        if area == "mixrmbac":
            return 20 if size <= 20 else 50 if size <= 50 else 225
        if area == "riomaior":
            return 317
        if area == "both":
            return 57 if size <= 57 else 371 if size <= 371 else 485 if size <= 485 else 542
        return size


def _ensure_directories(opts):
    """Ensure all required output and checkpoint directories exist."""
    try:
        parent_dir = os.path.join(
            udef.ROOT_DIR,
            "assets",
            opts["output_dir"],
            f"{opts['days']}_days",
            f"{opts['area']}_{opts['size']}",
        )
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(os.path.join(parent_dir, "fill_history", opts["data_distribution"]), exist_ok=True)
        os.makedirs(os.path.join(udef.ROOT_DIR, opts["checkpoint_dir"]), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, opts["checkpoint_dir"]), exist_ok=True)
    except Exception as e:
        raise Exception(
            f"directories to save WSR simulator test output files do not exist and could not be created: {e}"
        ) from e
