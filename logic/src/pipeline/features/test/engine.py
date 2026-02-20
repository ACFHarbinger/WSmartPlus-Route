"""
Simulation Test Runner.
"""

import os
import random
import re
from multiprocessing import cpu_count

import numpy as np
import torch

import logic.src.constants as udef
from logic.src.configs import Config
from logic.src.constants import MAP_DEPOTS, WASTE_TYPES
from logic.src.data.datasets import NumpyDictDataset, PandasExcelDataset
from logic.src.pipeline.simulations.repository import (
    FileSystemRepository,
    NumpyDictRepository,
    PandasExcelRepository,
    load_simulator_data,
    set_repository,
)

from .config import expand_policy_configs
from .orchestrator import simulator_testing


def run_wsr_simulator_test(cfg: Config) -> None:
    """
    Main entry point for the WSmart+ Route simulator test engine.

    This function orchestrates the end-to-end simulation testing workflow:
    1. Validates simulation configuration.
    2. Initializes random seeds for reproducibility.
    3. Resolves data availability and repository initialization.
    4. Expands policy configurations for multi-policy testing.
    5. Ensures output and checkpoint directories exist.
    6. Dispatches to the orchestrator for parallel or sequential execution.

    Args:
        cfg: Root Hydra configuration object containing ``cfg.sim`` with
            simulation parameters, policy settings, and environment metadata.
    """
    _validate_sim_config(cfg)

    sim = cfg.sim

    random.seed(sim.seed)
    np.random.seed(sim.seed)
    torch.manual_seed(sim.seed)

    data_size = _resolve_data_size(cfg)

    print(f"Area {sim.graph.area} ({data_size} available) for {sim.graph.num_loc} bins")
    if data_size != sim.graph.num_loc and not sim.graph.focus_graph:
        wtype_suffix = f"_{sim.graph.waste_type}" if sim.graph.waste_type else ""
        sim.graph.focus_graph = f"graphs_{sim.graph.area}_{sim.graph.num_loc}V_{sim.n_samples}N{wtype_suffix}.json"

    expand_policy_configs(cfg)
    _ensure_directories(cfg)

    device = torch.device("cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.device_count() - 1}")
    try:
        simulator_testing(cfg, data_size, device)
    except Exception as e:
        raise Exception(f"failed to execute WSmart+ Route simulations due to {repr(e)}") from e


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_sim_config(cfg: Config) -> None:
    """Validate and normalize ``cfg.sim`` fields in place."""
    sim = cfg.sim

    assert sim.days >= 1, "Must run the simulation for 1 or more days"
    assert sim.n_samples > 0, "Number of samples must be a positive integer"

    # Normalize area string (strip non-alpha, lowercase)
    sim.graph.area = re.sub(r"[^a-zA-Z]", "", sim.graph.area.lower())
    assert sim.graph.area in MAP_DEPOTS, f"Unknown area {sim.graph.area}, available areas: {list(MAP_DEPOTS.keys())}"

    # Normalize waste type
    sim.graph.waste_type = re.sub(r"[^a-zA-Z]", "", sim.graph.waste_type.lower())
    assert sim.graph.waste_type in WASTE_TYPES or sim.graph.waste_type is None, (
        f"Unknown waste type {sim.graph.waste_type}, available: {list(WASTE_TYPES.keys())}"
    )

    # Coerce edge_threshold to numeric
    sim.graph.edge_threshold = (
        float(sim.graph.edge_threshold) if "." in str(sim.graph.edge_threshold) else int(sim.graph.edge_threshold)
    )

    assert sim.cpu_cores >= 0, "Number of CPU cores must be >= 0"
    assert sim.cpu_cores <= cpu_count(), "Number of CPU cores to use cannot exceed system specifications"
    if sim.cpu_cores == 0:
        sim.cpu_cores = cpu_count()


# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------


def _resolve_data_size(cfg: Config) -> int:
    """Resolve the available data size for the given area and requested size."""
    sim = cfg.sim
    load_ds = cfg.load_dataset

    if load_ds is not None and str(load_ds).endswith(".npz"):
        dataset = NumpyDictDataset.load(load_ds)
        set_repository(NumpyDictRepository(dataset))
        _override_waste_filepath(cfg, load_ds)
        return sim.graph.num_loc

    if load_ds is not None and str(load_ds).endswith(".xlsx"):
        dataset = PandasExcelDataset.load(load_ds)
        set_repository(PandasExcelRepository(dataset))
        _override_waste_filepath(cfg, load_ds)
        return sim.graph.num_loc

    set_repository(FileSystemRepository(udef.ROOT_DIR))

    try:
        data_tmp, _ = load_simulator_data(sim.data_dir, sim.graph.num_loc, sim.graph.area, sim.graph.waste_type)
        return len(data_tmp)
    except Exception:
        area, size = sim.graph.area, sim.graph.num_loc
        if area == "mixrmbac":
            return 20 if size <= 20 else 50 if size <= 50 else 225
        if area == "riomaior":
            return 317
        if area == "both":
            return 57 if size <= 57 else 371 if size <= 371 else 485 if size <= 485 else 542
        return size


def _override_waste_filepath(cfg: Config, load_dataset: str) -> None:
    """Override ``sim.waste_filepath`` to point to *load_dataset*.

    ``Bins`` joins ``data_dir`` with ``waste_filepath``, so we compute the
    relative path from the simulator data root to the dataset file.
    """
    data_dir = os.path.join(udef.ROOT_DIR, "data", "wsr_simulator")
    abs_dataset = os.path.join(udef.ROOT_DIR, load_dataset) if not os.path.isabs(load_dataset) else load_dataset
    cfg.sim.waste_filepath = os.path.relpath(abs_dataset, data_dir)


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------


def _ensure_directories(cfg: Config) -> None:
    """Ensure all required output and checkpoint directories exist."""
    sim = cfg.sim
    try:
        parent_dir = os.path.join(
            udef.ROOT_DIR,
            "assets",
            sim.output_dir,
            f"{sim.days}_days",
            f"{sim.graph.area}_{sim.graph.num_loc}",
        )
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(
            os.path.join(parent_dir, "fill_history", sim.data_distribution),
            exist_ok=True,
        )
        os.makedirs(os.path.join(udef.ROOT_DIR, sim.checkpoint_dir), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, sim.checkpoint_dir), exist_ok=True)
    except Exception as e:
        raise Exception(
            f"directories to save WSR simulator test output files do not exist and could not be created: {e}"
        ) from e
