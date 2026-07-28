"""Simulation and data-generation controller.

Entry point for commands that exercise the simulation engine or the
dataset-generation pipeline, all routed through Hydra:

- ``test_sim``          — standard multi-day WSmart-Route simulator test
- ``hpo_sim`` / ``sim_hpo`` — hyperparameter optimisation over policy params
- ``gen_data``          — dataset generation (virtual / val / test splits)

Grouping rationale: all three commands operate on the *problem instance*
level rather than on the model weights.  ``gen_data`` produces the raw
data that the simulator (and trainer) consume; ``test_sim`` / ``hpo_sim``
run the full-stack routing simulation.

Example::

    python main.py test_sim sim.days=31 sim.policies=regular,gurobi,alns
    python main.py hpo_sim hpo_sim.policy=bpc hpo_sim.n_trials=20
    python main.py gen_data data.problem=vrpp data.graph_sizes=[50]
"""

from typing import Any, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_KEYS: List[str] = [
    "seed", "device", "experiment_name", "task",
    "output_dir", "run_name", "start", "tracking",
]

_SIM_TASKS = frozenset({"test_sim", "hpo_sim", "sim_hpo"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_config(cfg: Any, label: str, filter_keys: Any = None) -> None:
    """Pretty-print a filtered view of a Hydra config.

    Args:
        cfg: The Hydra configuration object.
        label: Section header label (e.g. ``"SIMULATION"``).
        filter_keys: Optional list of top-level keys to include.
    """
    from omegaconf import OmegaConf

    print("\n" + "=" * 80)
    print(f"HYDRA CONFIGURATION — {label}".center(80))
    print("=" * 80)
    display_cfg = OmegaConf.masked_copy(cfg, filter_keys) if filter_keys else cfg
    print(OmegaConf.to_yaml(display_cfg, resolve=False))
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Simulation main
# ---------------------------------------------------------------------------


def run_simulation(cfg: Any) -> float:
    """Run the WSmart-Route simulator or simulation HPO.

    Dispatches on ``cfg.task``:

    - ``"test_sim"``              → standard multi-day simulator test.
    - ``"hpo_sim"`` / ``"sim_hpo"`` → policy hyperparameter optimisation.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        0.0 on success.

    Raises:
        ValueError: If ``cfg.task`` is not a recognised simulation task.
    """
    task = cfg.task

    if task == "test_sim":
        from logic.src.pipeline.features.test import run_wsr_simulator_test

        if cfg.tracking.verbose:
            _print_config(cfg, "SIMULATION", filter_keys=_ROOT_KEYS + ["sim"])  # type: ignore[arg-type]
        run_wsr_simulator_test(cfg)
        return 0.0

    if task in ("hpo_sim", "sim_hpo"):
        from logic.src.pipeline.simulations.hpo.hpo_handler import run_hpo_sim

        if cfg.tracking.verbose:
            _print_config(cfg, "SIMULATION HPO", filter_keys=_ROOT_KEYS + ["hpo_sim"])  # type: ignore[arg-type]
        run_hpo_sim(cfg)
        return 0.0

    raise ValueError(f"Unknown simulation task: {task!r}")


# ---------------------------------------------------------------------------
# Data generation main
# ---------------------------------------------------------------------------


def run_data_generation(cfg: Any) -> float:
    """Generate datasets for training, validation, or testing.

    Initialises the WSmart+ tracking run, calls the dataset generator, and
    marks the run as completed regardless of outcome.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        0.0 on success.
    """
    import logic.src.tracking as wst
    from logic.src.data.generators import generate_datasets

    if cfg.tracking.verbose:
        _print_config(cfg, "DATA GENERATION", filter_keys=_ROOT_KEYS + ["data"])  # type: ignore[arg-type]

    experiment_name = cfg.experiment_name or f"gen_data_{cfg.data.problem}"
    wst.init(experiment_name=experiment_name)
    try:
        generate_datasets(cfg)
    finally:
        run = wst.get_active_run()
        if run is not None:
            run.set_tag("status", "completed")
            run.flush()
    return 0.0
