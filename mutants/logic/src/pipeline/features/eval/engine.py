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
from logic.src.pipeline.features.eval.evaluate import evaluate_policy, get_automatic_batch_size
from logic.src.utils.data.data_utils import save_dataset
from logic.src.utils.functions.function import load_model
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args
    model, _ = load_model(opts.get("load_path") or opts["model"])
    val_size = opts["val_size"] // num_processes
    dataset = model.problem.make_dataset(
        filename=dataset_path,
        num_samples=val_size,
        offset=opts["offset"] + val_size * i,
        distribution=opts["data_distribution"],
        vertex_strat=opts["vertex_method"],
        size=opts["graph_size"],
        focus_graph=opts["focus_graph"],
        focus_size=opts["focus_size"],
        area=opts["area"],
        dist_matrix_path=opts["dm_filepath"],
        number_edges=opts["edge_threshold"],
        waste_type=opts["waste_type"],
        dist_strat=opts["distance_method"],
        edge_strat=opts["edge_method"],
    )
    device = torch.device("cuda:{}".format(i))
    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def _eval_dataset(
    model: nn.Module,
    dataset: Dataset,
    width: int,
    softmax_temp: float,
    opts: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Inner evaluation loop for a single batch/process."""
    model.to(device)
    model.eval()
    if hasattr(model, "set_decode_type"):
        model.set_decode_type(
            "greedy" if opts["decode_strategy"] in ("bs", "greedy") else "sampling",
            temp=softmax_temp,
        )

    dataloader = DataLoader(dataset, batch_size=opts.get("eval_batch_size", 1024), pin_memory=True)
    results: List[Dict[str, Any]] = []

    method = opts.get("decode_strategy", "greedy")
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
            samples=width,
        )
        opts["eval_batch_size"] = eval_batch_size
        dataloader = DataLoader(dataset, batch_size=eval_batch_size, pin_memory=True)

    eval_results = evaluate_policy(
        model,
        model.problem,
        dataloader,
        method=method,
        return_results=True,
        samples=width,
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
                assert False, "Unknown problem: {}".format(model.problem.NAME)

        if seq is not None:
            seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)
            instance = dataset[i]
            if isinstance(instance, (list, tuple)):
                batch_i = {k: v.unsqueeze(0).to(device) for k, v in zip(["locs"], instance)}
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
    width: int,
    softmax_temp: float,
    opts: Dict[str, Any],
    method: Optional[str] = None,
) -> Tuple[List[float], List[Optional[List[int]]], List[float]]:
    """Evaluates a model on a given dataset."""
    model, _ = load_model(opts.get("load_path") or opts["model"])
    use_cuda = torch.cuda.is_available()
    results: List[Dict[str, Any]] = []

    if opts["multiprocessing"]:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts["val_size"] % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(
                itertools.chain.from_iterable(
                    pool.map(
                        eval_dataset_mp,
                        [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)],
                    )
                )
            )
    else:
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
            waste_type=opts["waste_type"],
            edge_strat=opts["edge_method"],
            dist_strat=opts["distance_method"],
        )
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    parallelism: int = opts["eval_batch_size"]
    avg_cost = np.mean([r["cost"] for r in results])
    std_cost = np.std([r["cost"] for r in results])
    avg_km = np.mean([r["km"] for r in results])
    avg_kg = np.mean([r["kg"] for r in results])
    avg_over = np.mean([r["overflows"] for r in results])

    logger.info("Average cost: {} +- {}".format(avg_cost, std_cost))
    logger.info("Average KM: {}, Average KG: {}, Average Overflows: {}".format(avg_km, avg_kg, avg_over))

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

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts["model"])[0]).split(os.sep)[-2:])
    if opts["output_filename"] is None:
        results_dir = os.path.join(opts["results_dir"], model.problem.NAME, dataset_basename)
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception:
            raise Exception("directories to save evaluation results do not exist and could not be created")

        out_file = os.path.join(
            results_dir,
            "{}-{}-{}{}-t{}-{}-{}{}".format(
                dataset_basename,
                model_name,
                opts["decode_strategy"],
                width if opts["decode_strategy"] != "greedy" else "",
                softmax_temp,
                opts["offset"],
                opts["offset"] + len(costs),
                ext,
            ),
        )
    else:
        out_file = opts["output_filename"]

    assert opts["overwrite"] or not os.path.isfile(
        out_file
    ), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)
    return costs, tours, durations
