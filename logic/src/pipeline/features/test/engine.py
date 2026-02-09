"""
Simulation Test Runner.
"""

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
    # Convert to standard dict for mutability and ease of use
    if not isinstance(opts, dict):
        try:
            opts = OmegaConf.to_container(opts, resolve=True)
        except Exception:
            pass  # Already a dict or compatible

    # Backwards compatibility: Map num_loc to size (main.py flattens graph config)
    if "num_loc" in opts:
        opts["size"] = opts["num_loc"]

    # If graph is present (not flattened), flatten it
    if "graph" in opts:
        opts["size"] = opts["graph"]["num_loc"]
        opts["area"] = opts["graph"]["area"]
        opts["waste_type"] = opts["graph"]["waste_type"]
        opts["dm_filepath"] = opts["graph"].get("dm_filepath")
        opts["waste_filepath"] = opts["graph"].get("waste_filepath")
        opts["edge_threshold"] = opts["graph"].get("edge_threshold")
        opts["edge_method"] = opts["graph"].get("edge_method")

    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    try:
        data_tmp, _ = load_simulator_data(opts.get("data_dir"), opts["size"], opts["area"], opts["waste_type"])
        data_size = len(data_tmp)
    except Exception:
        if opts["area"] == "mixrmbac" and opts["size"] not in [20, 50, 225]:
            data_size = 20 if opts["size"] < 20 else 50 if opts["size"] < 50 else 225
        elif opts["area"] == "riomaior" and opts["size"] not in [57, 104, 203, 317]:
            data_size = 317
        elif opts["area"] == "both" and opts["size"] not in [57, 371, 485, 542]:
            data_size = 57 if opts["size"] < 57 else 371 if opts["size"] < 371 else 485 if opts["size"] < 485 else 542
        else:
            data_size = opts["size"]

    print(f"Area {opts['area']} ({data_size} available) for {opts['size']} bins")
    if data_size != opts["size"] and (opts["bin_idx_file"] is None or opts["bin_idx_file"] == ""):
        opts["bin_idx_file"] = f"graphs_{opts['size']}V_{opts['n_samples']}N.json"

    expand_policy_configs(opts)

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
    except Exception:
        raise Exception("directories to save WSR simulator test output files do not exist and could not be created")

    device = torch.device("cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.device_count() - 1}")
    try:
        simulator_testing(opts, data_size, device)
    except Exception as e:
        raise Exception(f"failed to execute WSmart+ Route simulations due to {repr(e)}")
