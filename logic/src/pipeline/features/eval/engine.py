"""
Evaluation engine for WSmart-Route.
"""

import contextlib
import datetime
import itertools
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.pipeline.features.eval.evaluate import evaluate_policy, get_automatic_batch_size
from logic.src.utils.data.data_utils import save_dataset
from logic.src.utils.functions import load_model

mp = torch.multiprocessing.get_context("spawn")


def get_best(
    sequences: np.ndarray,
    cost: np.ndarray,
    ids: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[Optional[np.ndarray]], List[float]]:
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    """
    if ids is None:
        idx = cost.argmin()
        return [sequences[idx]], [float(cost[idx])]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)
    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]
    return [sequences[i] if i >= 0 else None for i in result], [float(cost[i]) if i >= 0 else math.inf for i in result]


def _build_dataset_kwargs(cfg: Config) -> Dict[str, Any]:
    """Build keyword arguments for make_dataset from Config."""
    ev = cfg.eval
    graph = ev.graph
    return {
        "num_samples": ev.val_size,
        "offset": ev.offset,
        "distribution": ev.data_distribution,
        "vertex_strat": graph.vertex_method,
        "size": graph.num_loc,
        "focus_graph": graph.focus_graph,
        "focus_size": graph.focus_size,
        "area": graph.area,
        "dist_matrix_path": graph.dm_filepath,
        "number_edges": graph.edge_threshold,
        "waste_weight": ev.reward.waste_weight if ev.reward else 1.0,
        "dist_strat": graph.distance_method,
        "edge_strat": graph.edge_method,
    }


def _build_eval_kwargs(cfg: Config) -> Dict[str, Any]:
    """Build keyword arguments for evaluate_policy from Config."""
    ev = cfg.eval
    result: Dict[str, Any] = {
        "eval_batch_size": ev.eval_batch_size,
        "no_progress_bar": cfg.tracking.no_progress_bar,
        "compress_mask": ev.compress_mask,
        "max_calc_batch_size": ev.max_calc_batch_size,
    }
    if ev.decoding:
        result["decoding"] = {
            "strategy": ev.decoding.strategy,
            "temperature": ev.decoding.temperature,
            "beam_width": ev.decoding.beam_width,
        }
    return result


def eval_dataset_mp(
    args: Tuple[str, int, float, Config, int, int],
) -> List[Dict[str, Any]]:
    """Worker function for multiprocessing evaluation."""
    (dataset_path, beam_width, softmax_temp, cfg, i, num_processes) = args
    ev = cfg.eval
    model_path = ev.policy.load_path if ev.policy else None
    model, _ = load_model(model_path)
    val_size = ev.val_size // num_processes

    ds_kwargs = _build_dataset_kwargs(cfg)
    ds_kwargs["num_samples"] = val_size
    ds_kwargs["offset"] = ev.offset + val_size * i

    dataset = model.problem.make_dataset(filename=dataset_path, **ds_kwargs)
    device = torch.device("cuda:{}".format(i))
    return _eval_dataset(model, dataset, beam_width, softmax_temp, cfg, device)


def _eval_dataset(
    model: nn.Module,
    dataset: Dataset,
    beam_width: int,
    softmax_temp: float,
    cfg: Config,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Inner evaluation loop for a single batch/process."""
    ev = cfg.eval
    strategy = ev.decoding.strategy if ev.decoding else "greedy"

    model.to(device)
    model.eval()
    if hasattr(model, "set_strategy"):
        model.set_strategy(strategy, temp=softmax_temp)

    eval_kwargs = _build_eval_kwargs(cfg)
    eval_batch_size = eval_kwargs.get("eval_batch_size", 1024)
    dataloader = DataLoader(dataset, batch_size=eval_batch_size, pin_memory=True)

    method = strategy
    if method == "sample":
        method = "sampling"

    # Automatic batch size tuning
    auto_batch = False  # Could be added to EvalConfig in the future
    if auto_batch:
        eval_batch_size = get_automatic_batch_size(
            model,
            model.problem,
            dataloader,
            method=method,
            initial_batch_size=eval_batch_size,
            samples=beam_width,
        )
        dataloader = DataLoader(dataset, batch_size=eval_batch_size, pin_memory=True)

    eval_results = evaluate_policy(
        model,
        model.problem,
        dataloader,
        method=method,
        return_results=True,
        samples=beam_width,
        softmax_temperature=softmax_temp,
        **eval_kwargs,
    )

    costs_best = eval_results["rewards"]
    sequences_best = eval_results["sequences"].cpu().numpy()
    duration_per_batch = eval_results["duration"] / len(dataloader)

    results: List[Dict[str, Any]] = []
    for i, (seq, cost) in enumerate(zip(sequences_best, costs_best)):
        if seq is not None:
            if model.problem.NAME in ("cvrpp", "cwcvrp", "sdwcvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]
            elif model.problem.NAME in ("vrpp", "wcvrp"):
                seq = np.trim_zeros(seq).tolist()
            else:
                seq = None
                raise AssertionError("Unknown problem: {}".format(model.problem.NAME))

        if seq is not None:
            seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)
            instance = dataset[i]
            if isinstance(instance, (list, tuple)):
                batch_i = {k: v.unsqueeze(0).to(device) for k, v in zip(["locs"], instance)}
            elif isinstance(instance, ITraversable):
                batch_i = {}
                for k, v in instance.items():
                    if torch.is_tensor(v):
                        batch_i[k] = v.unsqueeze(0).to(device)
                    elif isinstance(v, np.ndarray):
                        batch_i[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                    else:
                        batch_i[k] = v
            else:
                batch_i = instance.unsqueeze(0).to(device)

            _, c_dict, _ = model.problem.get_costs(batch_i, seq_tensor, None, batch_i.get("dist_matrix"))
        else:
            c_dict = {
                "length": torch.tensor(0.0),
                "waste": torch.tensor(0.0),
                "overflows": torch.tensor(0.0),
            }

        results.append(
            {
                "cost": float(cost),
                "seq": seq,
                "duration": duration_per_batch,
                "km": c_dict["length"].item(),
                "kg": c_dict["waste"].item() * 100,
                "overflows": c_dict["overflows"].item(),
            }
        )
    return results


def eval_dataset(
    dataset_path: str,
    beam_width: int,
    softmax_temp: float,
    cfg: Config,
    method: Optional[str] = None,
    strategy: Optional[str] = None,
    run: Optional[Any] = None,
) -> Tuple[List[float], List[Optional[List[int]]], List[float]]:
    """Evaluates a model on a given dataset.

    Args:
        dataset_path: Path to the evaluation dataset.
        beam_width: Beam width for beam-search decoding.
        softmax_temp: Softmax temperature for sampling.
        cfg: Root Hydra configuration.
        method: Decoding method override.
        strategy: Decoding strategy override.
        run: Optional WSTracker :class:`Run` for metric/artifact logging.

    Returns:
        Tuple of (costs, tours, durations).
    """
    ev = cfg.eval
    model_path = ev.policy.load_path if ev.policy else None
    model, _ = load_model(model_path)
    use_cuda = torch.cuda.is_available() and not getattr(ev, "no_cuda", False)
    if getattr(cfg, "device", None) == "cpu":
        use_cuda = False

    # Resolve strategy
    # resolved_strategy = strategy or method or (ev.decoding.strategy if ev.decoding else "greedy")

    if ev.multiprocessing:
        results = _eval_multiprocessing(dataset_path, beam_width, softmax_temp, cfg)
    else:
        results = _eval_singleprocess(model, dataset_path, beam_width, softmax_temp, cfg, use_cuda)

    parallelism: int = ev.eval_batch_size
    avg_metrics = {
        "cost": np.mean([r["cost"] for r in results]),
        "std": np.std([r["cost"] for r in results]),
        "km": np.mean([r["km"] for r in results]),
        "kg": np.mean([r["kg"] for r in results]),
        "overflows": np.mean([r["overflows"] for r in results]),
    }

    logger.info("Average cost: {} +- {}".format(avg_metrics["cost"], avg_metrics["std"]))
    logger.info(
        "Average KM: {}, Average KG: {}, Average Overflows: {}".format(
            avg_metrics["km"], avg_metrics["kg"], avg_metrics["overflows"]
        )
    )

    costs: List[float] = [float(r["cost"]) for r in results]
    tours: List[Optional[List[int]]] = [r["seq"] for r in results]
    durations: List[float] = [float(r["duration"]) for r in results]
    logger.info(
        "Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations)))
    )
    logger.info("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    logger.info(
        "Calculated total duration: {}".format(datetime.timedelta(seconds=int(np.sum(durations) / parallelism)))
    )

    model_name = "_".join(os.path.normpath(os.path.splitext(model_path or "model")[0]).split(os.sep)[-2:])
    out_file = ev.output_filename or _get_eval_output_path(
        model, dataset_path, cfg, model_name, beam_width, softmax_temp, len(costs)
    )

    assert ev.overwrite or not os.path.isfile(out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    # Log evaluation metrics and artifact to WSTracker run
    if run is not None:
        ds_name = os.path.splitext(os.path.basename(dataset_path))[0]
        prefix = f"eval/{ds_name}/bw{beam_width}"
        with contextlib.suppress(Exception):
            run.log_metric(f"{prefix}/avg_cost", float(avg_metrics["cost"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/std_cost", float(avg_metrics["std"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_km", float(avg_metrics["km"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_kg", float(avg_metrics["kg"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_overflows", float(avg_metrics["overflows"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_duration", float(np.mean(durations)))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/n_samples", float(len(costs)))  # type: ignore[arg-type]
        with contextlib.suppress(Exception):
            run.log_artifact(out_file, artifact_type="eval_results")
        with contextlib.suppress(Exception):
            run.log_dataset_event(  # type: ignore[union-attr]
                "load",
                file_path=dataset_path,
                num_samples=len(costs),
                metadata={
                    "event": "eval_dataset",
                    "beam_width": beam_width,
                    "strategy": strategy or method or "",
                },
            )

    return costs, tours, durations


def _eval_multiprocessing(dataset_path: str, beam_width: int, softmax_temp: float, cfg: Config) -> List[Dict[str, Any]]:
    """Helper for multiprocessing evaluation."""
    num_processes = torch.cuda.device_count()
    assert cfg.eval.val_size % num_processes == 0, "val_size must be divisible by num_processes"

    with mp.Pool(num_processes) as pool:
        return list(
            itertools.chain.from_iterable(
                pool.map(
                    eval_dataset_mp,
                    [(dataset_path, beam_width, softmax_temp, cfg, i, num_processes) for i in range(num_processes)],
                )
            )
        )


def _eval_singleprocess(
    model: nn.Module, dataset_path: str, beam_width: int, softmax_temp: float, cfg: Config, use_cuda: bool
) -> List[Dict[str, Any]]:
    """Helper for single-process evaluation."""
    device = torch.device("cpu" if not use_cuda else "cuda:0")

    ds_kwargs = _build_dataset_kwargs(cfg)
    dataset = model.problem.make_dataset(filename=dataset_path, **ds_kwargs)
    return _eval_dataset(model, dataset, beam_width, softmax_temp, cfg, device)


def _get_eval_output_path(
    model: nn.Module,
    dataset_path: str,
    cfg: Config,
    model_name: str,
    beam_width: int,
    softmax_temp: float,
    num_costs: int,
) -> str:
    """Generate evaluation result output path."""
    ev = cfg.eval
    strategy = ev.decoding.strategy if ev.decoding else "greedy"
    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    results_dir = os.path.join(ev.results_dir, model.problem.NAME, dataset_basename)
    os.makedirs(results_dir, exist_ok=True)

    return os.path.join(
        results_dir,
        "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename,
            model_name,
            strategy,
            beam_width if strategy != "greedy" else "",
            softmax_temp,
            ev.offset,
            ev.offset + num_costs,
            ext,
        ),
    )
