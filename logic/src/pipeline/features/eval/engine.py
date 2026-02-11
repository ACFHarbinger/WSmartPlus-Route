"""
Evaluation engine for WSmart-Route.
"""

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


def eval_dataset_mp(
    args: Tuple[str, int, float, Dict[str, Any], int, int],
) -> List[Dict[str, Any]]:
    """Worker function for multiprocessing evaluation."""
    (dataset_path, beam_width, softmax_temp, opts, i, num_processes) = args
    model, _ = load_model(opts.get("load_path") or opts["model"])
    val_size = opts["val_size"] // num_processes
    dataset = model.problem.make_dataset(
        filename=dataset_path,
        num_samples=val_size,
        offset=opts["offset"] + val_size * i,
        distribution=opts["data_distribution"],
        vertex_strat=opts["vertex_method"],
        size=opts.get("num_loc", 50),  # Default to 50 if graph_size not present
        focus_graph=opts["focus_graph"],
        focus_size=opts["focus_size"],
        area=opts["area"],
        dist_matrix_path=opts["dm_filepath"],
        number_edges=opts["edge_threshold"],
        waste_weight=opts.get("waste_weight", opts.get("w_waste", 1.0)),
        dist_strat=opts["distance_method"],
        edge_strat=opts["edge_method"],
    )
    device = torch.device("cuda:{}".format(i))
    return _eval_dataset(model, dataset, beam_width, softmax_temp, opts, device)


def _eval_dataset(
    model: nn.Module,
    dataset: Dataset,
    beam_width: int,
    softmax_temp: float,
    opts: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Inner evaluation loop for a single batch/process."""
    model.to(device)
    model.eval()
    if hasattr(model, "set_strategy"):
        model.set_strategy(
            opts["strategy"],
            temp=softmax_temp,
        )

    dataloader = DataLoader(dataset, batch_size=opts.get("eval_batch_size", 1024), pin_memory=True)
    results: List[Dict[str, Any]] = []

    method = opts.get("strategy", "greedy")
    if method == "sample":
        method = "sampling"

    # Automatic batch size tuning
    eval_batch_size = opts.get("eval_batch_size", 1024)
    if opts.get("auto_batch_size", False):
        eval_batch_size = get_automatic_batch_size(
            model,
            model.problem,
            dataloader,
            method=method,
            initial_batch_size=eval_batch_size,
            samples=beam_width,
        )
        opts["eval_batch_size"] = eval_batch_size
        dataloader = DataLoader(dataset, batch_size=eval_batch_size, pin_memory=True)

    eval_results = evaluate_policy(
        model,
        model.problem,
        dataloader,
        method=method,
        return_results=True,
        samples=beam_width,
        softmax_temperature=softmax_temp,
        **opts,
    )

    costs_best = eval_results["rewards"]
    sequences_best = eval_results["sequences"].cpu().numpy()
    duration_per_batch = eval_results["duration"] / len(dataloader)

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
    opts: Dict[str, Any],
    method: Optional[str] = None,
) -> Tuple[List[float], List[Optional[List[int]]], List[float]]:
    """Evaluates a model on a given dataset."""
    model, _ = load_model(opts.get("load_path") or opts["model"])
    use_cuda = torch.cuda.is_available()

    if opts["multiprocessing"]:
        results = _eval_multiprocessing(dataset_path, beam_width, softmax_temp, opts)
    else:
        results = _eval_singleprocess(model, dataset_path, beam_width, softmax_temp, opts, use_cuda)

    parallelism: int = opts["eval_batch_size"]
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

    model_name = "_".join(os.path.normpath(os.path.splitext(opts["model"])[0]).split(os.sep)[-2:])
    out_file = opts["output_filename"] or _get_eval_output_path(
        model, dataset_path, opts, model_name, beam_width, softmax_temp, len(costs)
    )

    assert opts["overwrite"] or not os.path.isfile(out_file), (
        "File already exists! Try running with -f option to overwrite."
    )

    save_dataset((results, parallelism), out_file)
    return costs, tours, durations


def _eval_multiprocessing(
    dataset_path: str, beam_width: int, softmax_temp: float, opts: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Helper for multiprocessing evaluation."""
    num_processes = torch.cuda.device_count()
    assert opts["val_size"] % num_processes == 0, "val_size must be divisible by num_processes"

    with mp.Pool(num_processes) as pool:
        return list(
            itertools.chain.from_iterable(
                pool.map(
                    eval_dataset_mp,
                    [(dataset_path, beam_width, softmax_temp, opts, i, num_processes) for i in range(num_processes)],
                )
            )
        )


def _eval_singleprocess(
    model: nn.Module, dataset_path: str, beam_width: int, softmax_temp: float, opts: Dict[str, Any], use_cuda: bool
) -> List[Dict[str, Any]]:
    """Helper for single-process evaluation."""
    device = torch.device("cpu" if not use_cuda else "cuda:0")
    dataset = model.problem.make_dataset(
        filename=dataset_path,
        num_samples=opts["val_size"],
        offset=opts["offset"],
        distribution=opts["data_distribution"],
        vertex_strat=opts["vertex_method"],
        size=opts["graph_size"],
        focus_graph=opts["focus_graph"],
        focus_size=opts["focus_size"],
        area=opts["area"],
        number_edges=opts["edge_threshold"],
        dist_matrix_path=opts["dm_filepath"],
        waste_weight=opts.get("waste_weight", opts.get("w_waste", 1.0)),
        edge_strat=opts["edge_method"],
        dist_strat=opts["distance_method"],
    )
    return _eval_dataset(model, dataset, beam_width, softmax_temp, opts, device)


def _get_eval_output_path(
    model: nn.Module,
    dataset_path: str,
    opts: Dict[str, Any],
    model_name: str,
    beam_width: int,
    softmax_temp: float,
    num_costs: int,
) -> str:
    """Generate evaluation result output path."""
    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    results_dir = os.path.join(opts["results_dir"], model.problem.NAME, dataset_basename)
    os.makedirs(results_dir, exist_ok=True)

    return os.path.join(
        results_dir,
        "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename,
            model_name,
            opts["strategy"],
            beam_width if opts["strategy"] != "greedy" else "",
            softmax_temp,
            opts["offset"],
            opts["offset"] + num_costs,
            ext,
        ),
    )
