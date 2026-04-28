"""
Core handler for simulation policy hyperparameter optimization (HPO).

This module implements the Optuna objective function for simulation policies,
including parameter sampling, simulation execution, and metric calculation.
It supports process-based parallel optimization via a shared SQLite storage
backend, multi-objective Pareto-front search, Hyperband pruning, and
integrates with the MLflow tracking system.

Key improvements over the previous version:
  - Sampler method is actually used (was previously ignored).
  - Study is persisted to SQLite so crashed runs can be resumed.
  - Parallel execution uses separate OS processes + shared storage,
    avoiding GIL contention on CPU-bound simulations.
  - Multi-objective mode (e.g. profit ↑ + overflows ↓) supported via NSGA-II.
  - Hyperband pruner for iterative solvers.
  - Parameter suggestion is fully delegated to PolicyHPOBase.suggest_param
    (no duplicated inline logic).
  - Param names passed to Optuna are always the short name; the full config
    path is tracked in a separate PARAM_PATH_MAP per trial to keep the study
    DB coherent across restarts.
  - Search space is validated before any trial runs.

Attributes:
    logger: Logger instance for HPO operations.

Functions:
    objective:     Optuna objective for a single trial (single or multi-objective).
    worker:        Entry point for each worker process.
    run_hpo_sim:   Orchestrates the parallel HPO process.

Example:
    >>> # from logic.src.policies.helpers.hpo import run_hpo_sim
    >>> # best_metric = run_hpo_sim(cfg)
    >>> # print(f"Best metric: {best_metric}")
"""

import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import optuna
import torch
from omegaconf import OmegaConf

from logic.src import tracking as wst
from logic.src.configs import Config
from logic.src.constants import ROOT_DIR, SIM_METRICS
from logic.src.pipeline.features.test.config import expand_policy_configs
from logic.src.pipeline.simulations.hpo.base import PolicyHPOBase
from logic.src.pipeline.simulations.hpo.search_spaces import validate_search_space
from logic.src.pipeline.simulations.repository import (
    load_indices,
    load_simulator_data,
    set_repository_from_path,
)
from logic.src.pipeline.simulations.simulator import sequential_simulations
from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum recommended number of simulation samples per trial.
# A single sample has high variance for stochastic solvers (ALNS, ACO, etc.).
MIN_RECOMMENDED_SAMPLES = 5

# Metrics that are minimised (lower is better); all others are maximised.
_MINIMISE_METRICS = {"overflows", "kg_lost", "length"}


def _metric_direction(metric_name: str) -> str:
    """Return the Optuna direction string for a given metric name.

    Args:
        metric_name (str): Metric identifier (e.g. 'profit', 'overflows').

    Returns:
        str: 'minimize' or 'maximize'.
    """
    return "minimize" if metric_name in _MINIMISE_METRICS else "maximize"


def _extract_metric(log: Dict, metric_name: str) -> float:
    """Extract a scalar metric value from a simulation log dict.

    Args:
        log (Dict): Output of sequential_simulations keyed by policy name.
        metric_name (str): Metric to extract (must be in SIM_METRICS).

    Returns:
        float: Metric value, or -inf / +inf depending on direction when
            the log is empty or the metric is not found.
    """
    direction = _metric_direction(metric_name)
    fallback = float("inf") if direction == "minimize" else float("-inf")

    if metric_name in SIM_METRICS:
        idx = SIM_METRICS.index(metric_name)
    elif "profit" in SIM_METRICS:
        idx = SIM_METRICS.index("profit")
    else:
        return fallback

    pol_name = list(log.keys())[0] if log else None
    if pol_name and log[pol_name]:
        return float(log[pol_name][idx])

    return fallback


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(  # noqa: C901
    trial: optuna.Trial,
    base_cfg: Config,
    data_size: int,
    lock: Any,
) -> Union[float, Tuple[float, ...]]:
    """Optuna objective function for simulation policy HPO.

    Supports both single-objective and multi-objective modes, controlled by
    ``base_cfg.hpo_sim.metrics`` (a list of metric names).  When the list has
    exactly one entry the function returns a scalar; otherwise it returns a
    tuple so Optuna can compute the Pareto front.

    Parameter names passed to ``trial.suggest_*`` are always the *short* names
    from the search space definition (without the full config path prefix).
    The mapping to config paths is stored in ``trial.set_user_attr`` for
    auditability and study resumption safety.

    Args:
        trial (optuna.Trial): The Optuna trial object for parameter sampling.
        base_cfg (Config): The base configuration; cloned per trial.
        data_size (int): Number of available bins for the selected area.
        lock (Any): Multiprocessing lock for repository access.

    Returns:
        Union[float, Tuple[float, ...]]: Metric value(s) for this trial.
    """
    hpo_sim = base_cfg.hpo_sim
    policy_name = hpo_sim.policy_name

    # Resolve metric list (backwards-compatible with scalar 'metric' field).
    metrics: List[str] = list(getattr(hpo_sim, "metrics", None) or [getattr(hpo_sim, "metric", "profit")])
    search_space: Dict[str, Any] = dict(hpo_sim.search_space)

    # -----------------------------------------------------------------
    # 1. Build param→config-path mapping and sample from search space.
    # -----------------------------------------------------------------
    # Short names are used with trial.suggest_* so Optuna's internal state
    # (which drives TPE/CMA-ES surrogate models) is consistent across study
    # resumptions.  The full config path is stored as a trial attribute.
    param_path_map: Dict[str, str] = {}
    params_by_path: Dict[str, Any] = {}

    for short_name, spec in search_space.items():
        full_path = short_name if "." in short_name else f"sim.full_policies.0.{policy_name}.{short_name}"
        param_path_map[short_name] = full_path

        value = PolicyHPOBase.suggest_param(trial, short_name, spec)
        params_by_path[full_path] = value

    # Store the mapping for auditability (visible in Optuna dashboard / DB).
    trial.set_user_attr("param_path_map", param_path_map)

    # -----------------------------------------------------------------
    # 2. Clone config and apply sampled params.
    # -----------------------------------------------------------------
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    trial_cfg = OmegaConf.create(cfg_dict)

    # Ensure full_policies[0] contains the target policy as a dict.
    if not trial_cfg.sim.full_policies:
        trial_cfg.sim.full_policies = [{policy_name: {}}]
    elif (
        isinstance(trial_cfg.sim.full_policies[0], str)
        or isinstance(trial_cfg.sim.full_policies[0], dict)
        and policy_name not in trial_cfg.sim.full_policies[0]
    ):
        trial_cfg.sim.full_policies[0] = {policy_name: {}}

    for path, val in params_by_path.items():
        OmegaConf.update(trial_cfg, path, val)

    # -----------------------------------------------------------------
    # 3. Synchronise graph / simulation settings.
    # -----------------------------------------------------------------
    trial_cfg.sim.graph.num_loc = hpo_sim.graph.num_loc
    trial_cfg.sim.graph.area = hpo_sim.graph.area
    trial_cfg.sim.graph.waste_type = hpo_sim.graph.waste_type
    trial_cfg.sim.days = hpo_sim.graph.n_days
    trial_cfg.sim.n_samples = hpo_sim.graph.n_samples

    if hasattr(trial_cfg.sim.graph, "edge_threshold"):
        try:
            trial_cfg.sim.graph.edge_threshold = int(trial_cfg.sim.graph.edge_threshold)
        except (ValueError, TypeError):
            trial_cfg.sim.graph.edge_threshold = 0

    trial_cfg.sim.policies = [policy_name]
    expand_policy_configs(trial_cfg)  # type: ignore[arg-type]

    # -----------------------------------------------------------------
    # 4. Resolve instance indices.
    # -----------------------------------------------------------------
    n_samples = hpo_sim.graph.n_samples
    num_loc = hpo_sim.graph.num_loc

    if data_size != num_loc:
        focus_graph = trial_cfg.sim.graph.focus_graph
        if not focus_graph:
            wtype_suffix = f"_{hpo_sim.graph.waste_type}" if hpo_sim.graph.waste_type else ""
            focus_graph = f"graphs_{hpo_sim.graph.area}_{num_loc}V_{n_samples}N{wtype_suffix}.json"

        indices = load_indices(focus_graph, n_samples, num_loc, data_size, lock)
        if len(indices) == 1:
            indices = [indices[0]] * n_samples
    else:
        indices = [None] * n_samples  # type: ignore[assignment]

    sample_idx_ls = [list(range(n_samples))]
    device = torch.device(getattr(trial_cfg, "device", "cpu"))

    # -----------------------------------------------------------------
    # 5. Run simulations.
    # -----------------------------------------------------------------
    directions = [_metric_direction(m) for m in metrics]
    # Fallback values: worst possible for each direction.
    fallbacks = tuple(float("inf") if d == "minimize" else float("-inf") for d in directions)

    try:
        log, _, _ = sequential_simulations(
            cfg=trial_cfg,  # type: ignore[arg-type]
            device=device,
            indices_ls=indices,
            sample_idx_ls=sample_idx_ls,
            model_weights_path="",
            lock=lock,
        )

        values = tuple(_extract_metric(log, m) for m in metrics)

    except optuna.TrialPruned:
        raise  # Propagate pruning signal correctly.

    except Exception as exc:
        logger.error(f"Trial {trial.number} failed: {exc}")
        values = fallbacks

    # -----------------------------------------------------------------
    # 6. Return scalar or tuple depending on objective mode.
    # -----------------------------------------------------------------
    return values[0] if len(values) == 1 else values


# ---------------------------------------------------------------------------
# Worker (process-based parallelism)
# ---------------------------------------------------------------------------


def worker(
    study_name: str,
    storage_url: str,
    base_cfg_yaml: str,
    data_size: int,
    n_trials: int,
    lock: Any,
) -> None:
    """Entry point for each HPO worker process.

    Each process loads the shared Optuna study from the persistent storage
    backend and independently contributes trials.  This avoids GIL contention
    entirely because each process runs its own Python interpreter.

    Args:
        study_name (str): Name of the Optuna study to load.
        storage_url (str): SQLAlchemy URL for the shared storage backend.
        base_cfg_yaml (str): Serialised OmegaConf YAML of the base Config.
            Passed as a string so the Config can be reconstructed inside the
            forked process without pickling issues.
        data_size (int): Number of available bins for the selected area.
        n_trials (int): Number of trials this worker should contribute.
        lock (Any): Shared multiprocessing lock for repository access.
    """
    # Silence per-worker Optuna INFO logs to avoid interleaved console noise.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_cfg = OmegaConf.create(base_cfg_yaml)

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(
        lambda trial: objective(trial, base_cfg, data_size, lock),
        n_trials=n_trials,
        catch=(Exception,),  # Prevent a single bad trial from killing the worker.
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_hpo_sim(cfg: Config) -> Union[float, List[float]]:
    """Run Hyperparameter Optimization for simulation policies.

    Orchestrates the full HPO pipeline:
      1. Validates the search space before committing any resources.
      2. Creates or resumes a persistent Optuna study.
      3. Launches worker processes (one per ``num_workers``), each contributing
         an equal share of trials via the shared storage backend.
      4. Logs the best result(s) to the experiment tracker.

    In multi-objective mode (``hpo_sim.metrics`` has >1 entry), returns the
    list of objective values for the Pareto-optimal trial closest to the
    ideal point; in single-objective mode, returns the best scalar value.

    Args:
        cfg (Config): Root application configuration containing HPO settings.

    Returns:
        Union[float, List[float]]: Best metric value (single-objective) or
            best Pareto-front objective values (multi-objective).
    """
    hpo_sim = cfg.hpo_sim
    sim = cfg.sim

    # -----------------------------------------------------------------
    # 0. Resolve and validate search space eagerly.
    # -----------------------------------------------------------------
    search_space = dict(hpo_sim.search_space)

    if not search_space:
        from logic.src.pipeline.simulations.hpo.search_spaces import compose_search_space

        search_space = compose_search_space(
            job=hpo_sim.policy_name,
            filter=getattr(hpo_sim, "selection_name", None),
            interceptor=getattr(hpo_sim, "improver_name", None),
            rule=getattr(hpo_sim, "acceptance_name", None),
        )

        if not search_space:
            logger.warning(
                f"No search space provided in config and no default found for "
                f"policy='{hpo_sim.policy_name}'. HPO will perform no parameter changes."
            )
    else:
        # Validate the manually provided search space.
        validate_search_space(search_space, hpo_sim.policy_name)

    # Warn if n_samples is too low for reliable HPO.
    n_samples = hpo_sim.graph.n_samples
    if n_samples < MIN_RECOMMENDED_SAMPLES:
        logger.warning(
            f"hpo_sim.graph.n_samples={n_samples} is below the recommended minimum of "
            f"{MIN_RECOMMENDED_SAMPLES}. Results may be noisy. Consider increasing n_samples."
        )

    # -----------------------------------------------------------------
    # 1. Resolve metric list (supports legacy scalar 'metric' field).
    # -----------------------------------------------------------------
    metrics: List[str] = list(getattr(hpo_sim, "metrics", None) or [getattr(hpo_sim, "metric", "profit")])
    directions = [_metric_direction(m) for m in metrics]
    is_multi_objective = len(metrics) > 1

    logger.info(f"Metrics: {metrics}  |  Directions: {directions}")

    if is_multi_objective and hpo_sim.method.lower() not in ("nsgaii", "random"):
        logger.warning(
            f"Multi-objective HPO with method='{hpo_sim.method}' may not work correctly. "
            f"Consider switching to method='nsgaii'."
        )

    # -----------------------------------------------------------------
    # 2. Initialise experiment tracking.
    # -----------------------------------------------------------------
    experiment_name = cfg.experiment_name or f"hpo_sim_{hpo_sim.policy_name}"
    wst.init(experiment_name=experiment_name)

    # -----------------------------------------------------------------
    # 3. Resolve data repository and data_size.
    # -----------------------------------------------------------------
    load_ds = getattr(cfg, "load_dataset", None)
    if load_ds is not None and set_repository_from_path(str(load_ds)):
        data_size = sim.graph.num_loc
    else:
        set_repository_from_path(str(ROOT_DIR))
        try:
            data_tmp, _ = load_simulator_data(sim.data_dir, sim.graph.num_loc, sim.graph.area, sim.graph.waste_type)
            data_size = len(data_tmp)
        except Exception:
            data_size = sim.graph.num_loc

    # -----------------------------------------------------------------
    # 4. Build sampler and pruner.
    # -----------------------------------------------------------------
    sampler = PolicyHPOBase.build_sampler(
        method=hpo_sim.method,
        seed=cfg.seed,
        search_space=search_space,
    )

    # Hyperband pruner: starts pruning after min_resource steps, scales up
    # to n_days (each day is a natural evaluation checkpoint for the WCRP).
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=hpo_sim.graph.n_days,
        reduction_factor=3,
    )

    # -----------------------------------------------------------------
    # 5. Create or resume the persistent study.
    # -----------------------------------------------------------------
    output_dir = getattr(cfg, "output_dir", str(ROOT_DIR))
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, f"hpo_{hpo_sim.policy_name}.db")
    storage_url = f"sqlite:///{db_path}"
    study_name = f"{hpo_sim.policy_name}_seed{cfg.seed}"

    logger.info(f"HPO storage: {db_path}  (study='{study_name}')")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,  # Resume if the study already exists.
        directions=directions,
        sampler=sampler,
        pruner=pruner,
    )

    # -----------------------------------------------------------------
    # 6. Launch worker processes.
    # -----------------------------------------------------------------
    n_workers = max(1, hpo_sim.num_workers)
    n_trials = hpo_sim.n_trials
    trials_per_worker = max(1, n_trials // n_workers)
    # The last worker picks up any remainder from integer division.
    trials_last_worker = n_trials - trials_per_worker * (n_workers - 1)

    logger.info(
        f"Starting HPO  policy={hpo_sim.policy_name}  method={hpo_sim.method}  trials={n_trials}  workers={n_workers}"
    )

    # Serialise the config to YAML for safe inter-process transmission.
    base_cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)

    manager = mp.Manager()
    lock = manager.Lock()

    if n_workers == 1:
        # Single-worker: run in-process to keep the stack trace readable.
        worker(study_name, storage_url, base_cfg_yaml, data_size, n_trials, lock)
    else:
        processes = []
        for i in range(n_workers):
            worker_trials = trials_last_worker if i == n_workers - 1 else trials_per_worker
            p = mp.Process(
                target=worker,
                args=(study_name, storage_url, base_cfg_yaml, data_size, worker_trials, lock),
                daemon=False,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # Reload study from storage to get updated trials from all workers.
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # -----------------------------------------------------------------
    # 7. Extract best result(s) and log to tracker.
    # -----------------------------------------------------------------
    run = wst.get_active_run()

    if is_multi_objective:
        pareto_trials = study.best_trials
        logger.info(f"HPO complete. Pareto front size: {len(pareto_trials)}")

        # Report the trial closest to the ideal point (normalised L1 distance).
        best_trial = _select_pareto_representative(pareto_trials, directions)
        best_values = list(best_trial.values) if best_trial else []

        if run is not None and best_trial:
            run.log_params({f"hpo/best/{k}": v for k, v in best_trial.params.items()})
            for m, v in zip(metrics, best_values, strict=False):
                run.log_metric(f"hpo/best/{m}", v)
            run.set_tag("task", "hpo_sim")
            run.set_tag("policy", hpo_sim.policy_name)
            run.set_tag("mode", "multi_objective")
            run.flush()

        return best_values

    else:
        best_trial = study.best_trial
        best_value = best_trial.value if best_trial else float("-inf")

        logger.info(f"HPO complete. Best {metrics[0]}: {best_value}")
        logger.info(f"Best params: {best_trial.params if best_trial else {}}")

        if run is not None and best_trial:
            run.log_params({f"hpo/best/{k}": v for k, v in best_trial.params.items()})
            run.log_metric(f"hpo/best_{metrics[0]}", best_value)
            run.set_tag("task", "hpo_sim")
            run.set_tag("policy", hpo_sim.policy_name)
            run.set_tag("mode", "single_objective")
            run.flush()

        return best_value


# ---------------------------------------------------------------------------
# Pareto selection helper
# ---------------------------------------------------------------------------


def _select_pareto_representative(
    pareto_trials: List[optuna.trial.FrozenTrial],
    directions: List[str],
) -> Optional[optuna.trial.FrozenTrial]:
    """Select a single representative trial from a Pareto front.

    Uses a normalised L1 distance to the ideal point (best value per
    objective across all Pareto trials).  Minimisation objectives are
    negated before distance computation so that 'closer to ideal'
    always means 'smaller L1 distance'.

    Args:
        pareto_trials (List[optuna.trial.FrozenTrial]): Trials on the Pareto front.
        directions (List[str]): Per-objective directions ('minimize'/'maximize').

    Returns:
        Optional[optuna.trial.FrozenTrial]: The representative trial, or None
            if pareto_trials is empty.
    """
    if not pareto_trials:
        return None
    if len(pareto_trials) == 1:
        return pareto_trials[0]

    n_obj = len(directions)

    # Collect objective values, sign-flipped so that 'more positive = better'.
    signed: List[List[float]] = []
    for t in pareto_trials:
        row = []
        for _j, (v, d) in enumerate(zip(t.values, directions, strict=False)):
            row.append(-v if d == "minimize" else v)
        signed.append(row)

    # Ideal point: max per objective in signed space.
    ideal = [max(row[j] for row in signed) for j in range(n_obj)]
    nadir = [min(row[j] for row in signed) for j in range(n_obj)]
    ranges = [max(ideal[j] - nadir[j], 1e-9) for j in range(n_obj)]

    # Normalised L1 distance from each trial to the ideal point.
    distances = [sum(abs(ideal[j] - row[j]) / ranges[j] for j in range(n_obj)) for row in signed]

    best_idx = distances.index(min(distances))
    return pareto_trials[best_idx]
