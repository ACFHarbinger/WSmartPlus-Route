"""
Simulation Test Runner.
"""

import contextlib
import os
import random
import re
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import logic.src.constants as udef
import logic.src.tracking as wst
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
from logic.src.tracking.logging.pylogger import get_pylogger

from .config import expand_policy_configs
from .orchestrator import simulator_testing

logger = get_pylogger(__name__)


def run_wsr_simulator_test(cfg: Config, sinks: Optional[List[Any]] = None) -> None:
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
        sinks: Optional list of tracking sinks (e.g. :class:`ZenMLBridge`)
            to attach to the WSTracker run. When ``None`` (the default),
            :class:`MLflowBridge` is auto-attached if
            ``cfg.tracking.mlflow_enabled`` is ``True``.
    """
    # ----- ZenML dispatch (opt-in) -----
    tracking = getattr(cfg, "tracking", None)
    zenml_enabled = bool(getattr(tracking, "zenml_enabled", False))
    if zenml_enabled and sinks is None:
        _run_sim_via_zenml(cfg)
        return

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

    # --- Centralised experiment tracking ---
    experiment_name = f"sim-{sim.graph.area}-{sim.graph.num_loc}bins-{sim.days}days"
    tracker = wst.init(experiment_name=experiment_name)
    policy_names = [str(p) if not isinstance(p, dict) else list(p.keys())[0] for p in sim.full_policies]
    run_tags = {
        "area": sim.graph.area,
        "num_loc": str(sim.graph.num_loc),
        "days": str(sim.days),
        "n_samples": str(sim.n_samples),
        "policies": ",".join(policy_names),
        "seed": str(sim.seed),
        "distribution": str(sim.data_distribution),
    }
    # The run is entered as a context manager so get_active_run() is set for
    # all code executing in this process (including sequential simulation workers).
    # For parallel (spawned) workers the URI + run_id are threaded through
    # initargs by the orchestrator, which reads them from the active tracker.
    run = tracker.start_run(experiment_name, run_type="simulation", tags=run_tags)
    run.__enter__()

    # Log full simulation config and per-policy settings
    _log_sim_params(run, cfg)

    # ----- Attach secondary sinks -----
    if sinks is not None:
        for sink in sinks:
            run.add_sink(sink)
    else:
        # Auto-attach MLflowBridge when running outside ZenML
        mlflow_enabled = bool(getattr(tracking, "mlflow_enabled", False))
        if mlflow_enabled:
            with contextlib.suppress(Exception):
                wst.MLflowBridge.attach(
                    run,
                    mlflow_tracking_uri=str(getattr(tracking, "mlflow_tracking_uri", "mlruns")),
                    experiment_name=str(getattr(tracking, "mlflow_experiment_name", "wsmart-route")),
                    run_name=run.run_id[:8],
                    tags=run_tags,
                )

    # Log simulation data directory baseline hashes for change detection
    try:
        data_dir = os.path.join(udef.ROOT_DIR, "data", "wsr_simulator")
        if os.path.isdir(data_dir):
            wst.FilesystemTracker(run).scan_directory(data_dir)
    except Exception:
        pass

    try:
        simulator_testing(cfg, data_size, device)
        run.__exit__(None, None, None)
    except Exception as e:
        run.__exit__(type(e), e, e.__traceback__)
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


# ---------------------------------------------------------------------------
# Parameter logging
# ---------------------------------------------------------------------------


def _log_sim_params(run: wst.Run, cfg: Config) -> None:
    """Log simulation configuration and per-policy settings as run parameters.

    Captures the full ``cfg.sim`` scalar fields and each policy's config dict
    so that every run is fully reproducible from the tracking DB alone.

    Args:
        run: Active WSTracker run to receive the parameters.
        cfg: Root Hydra config object.
    """
    sim = cfg.sim

    params: Dict[str, Any] = {
        "sim.days": sim.days,
        "sim.n_samples": sim.n_samples,
        "sim.seed": sim.seed,
        "sim.data_distribution": str(sim.data_distribution),
        "sim.area": sim.graph.area,
        "sim.num_loc": sim.graph.num_loc,
        "sim.waste_type": sim.graph.waste_type,
        "sim.cpu_cores": sim.cpu_cores,
        "sim.checkpoint_dir": str(getattr(sim, "checkpoint_dir", "")),
        "sim.output_dir": str(getattr(sim, "output_dir", "")),
    }

    # Per-policy configuration
    for pol in getattr(sim, "full_policies", []):
        if isinstance(pol, dict):
            for name, pol_cfg in pol.items():
                with contextlib.suppress(Exception):
                    if hasattr(pol_cfg, "items"):
                        for k, v in pol_cfg.items():
                            params[f"policy.{name}.{k}"] = v
                    else:
                        params[f"policy.{name}"] = str(pol_cfg)
        else:
            params[f"policy.{pol}"] = True

    # External dataset path (if provided)
    load_ds = getattr(cfg, "load_dataset", None)
    if load_ds:
        params["load_dataset"] = str(load_ds)
        with contextlib.suppress(Exception):
            run.log_dataset_event(
                "load",
                file_path=str(load_ds),
                metadata={
                    "variable_name": "load_dataset",
                    "source_file": "features/test/engine.py",
                    "source_line": 297,
                },
            )

    run.log_params(params)


# ---------------------------------------------------------------------------
# ZenML dispatch
# ---------------------------------------------------------------------------


def _run_sim_via_zenml(cfg: Config) -> None:
    """Dispatch simulation testing to the ZenML simulation pipeline.

    Called when ``cfg.tracking.zenml_enabled`` is ``True`` and no external
    sinks were injected (i.e. the call originates from the CLI).
    """
    tracking = getattr(cfg, "tracking", None)
    mlflow_uri = str(getattr(tracking, "mlflow_tracking_uri", "mlruns"))
    stack_name = str(getattr(tracking, "zenml_stack_name", "wsmart-route-stack"))

    from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack

    if not configure_zenml_stack(mlflow_uri, stack_name=stack_name):
        logger.warning("ZenML stack configuration failed — falling back to direct simulation.")
        run_wsr_simulator_test(cfg, sinks=[])
        return

    try:
        from logic.src.pipeline.features.test.zenml_sim_pipeline import simulation_pipeline

        simulation_pipeline(cfg)
    except Exception as exc:
        logger.warning(f"ZenML simulation pipeline failed — falling back to direct simulation: {exc}")
        run_wsr_simulator_test(cfg, sinks=[])
