"""
Epoch Management Utilities for REINFORCE Training.

This module provides utility functions for managing training epochs, validation,
dataset preparation, and model updates during the REINFORCE training loop.

Key responsibilities:
- Dataset initialization and time-based updates
- Validation rollouts and metric computation
- Gradient clipping and normalization
- Batch preparation and preprocessing
- Model state management (DataParallel unwrapping)

These functions are used by trainer classes to orchestrate the training flow
while keeping the core logic modular and reusable.
"""

import json
import math
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from logic.src.utils.data_utils import generate_waste_prize
from logic.src.utils.functions.function import get_inner_model, move_to

from ...simulations.bins import Bins


def set_decode_type(model, decode_type):
    """
    Set the decoding strategy for the model.

    This function handles DataParallel-wrapped models by accessing the underlying module.

    Args:
        model: Neural model (potentially wrapped in DataParallel)
        decode_type: String specifying decode type ('greedy', 'sampling', 'beam_search')
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def rollout(model, dataset, opts):
    """
    Perform a complete rollout over a dataset.

    Evaluates the model on all instances in the dataset using greedy decoding.
    Handles temporal models (TAM) by initializing fill history if needed.

    Args:
        model: Neural model to evaluate
        dataset: Dataset to roll out on
        opts: Options dictionary with configuration parameters

    Returns:
        torch.Tensor: Costs for all instances in the dataset [num_instances]
    """

    def _eval_model_bat(bat):
        with torch.no_grad():
            ucost, _, _, _, _ = model(move_to(bat, opts["device"], non_blocking=True), cost_weights=None)
        return ucost.data.cpu()

    set_decode_type(model, "greedy")
    model.eval()
    if opts.get("temporal_horizon", 0) > 0 and opts.get("model") in ["tam"]:
        dataset.fill_history = torch.zeros((opts["val_size"], opts["graph_size"], opts["temporal_horizon"]))
        dataset.fill_history[:, :, -1] = torch.stack(
            [
                instance.get("waste", instance.get("noisy_waste", instance.get("real_waste")))
                for instance in dataset.data
            ]
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts["eval_batch_size"],
        pin_memory=True,
        num_workers=opts.get("num_workers", 0),
    )
    costs = []
    for bat_id, bat in enumerate(tqdm(dataloader, disable=opts.get("no_progress_bar", True))):
        bat = prepare_batch(bat, bat_id, dataset, dataloader, opts)
        cost = _eval_model_bat(bat)
        costs.append(cost)
    return torch.cat(costs, 0)


def validate_update(model, dataset, opts, cw_dict=None, metric=None, dist_matrix=None):
    """
    Validate the model, compute rich metrics, and optionally update cost weights or return HPO scores.

    This unified function handles:
    1. Simple validation (returns avg_cost)
    2. Adaptive weight updates for Meta-Learning (returns new_cw, avg_cost, all_costs)
    3. HPO metric calculation (returns metric_score, mean_ucost, all_costs)

    Args:
        model: Neural model.
        dataset: Validation dataset.
        opts (dict): Options dictionary.
        cw_dict (dict, optional): Current cost function weights for adaptive update.
        metric (str, optional): Metric name for HPO scoring ('overflows', 'kg/km', 'both').
        dist_matrix (torch.Tensor, optional): Distance matrix for HPO validation.

    Returns:
        Varies based on arguments:
        - if cw_dict: (new_cw, avg_cost, all_costs)
        - elif metric: (metric_score, avg_cost, all_costs)
        - else: avg_cost
    """
    from logic.src.policies.neural_agent import NeuralAgent

    agent = NeuralAgent(get_inner_model(model))

    def _eval_model_bat(bat, d_matrix):
        if d_matrix is None:
            # Fallback for simple validation without distance matrix
            with torch.no_grad():
                ucost, _, c_dict, _, _ = model(move_to(bat, opts["device"], non_blocking=True), cost_weights=None)

            # Map cost dictionary keys to NeuralAgent output format
            ret_dict = {}
            if c_dict is not None:
                ret_dict["waste"] = c_dict.get("waste")
                # Approximate 'km' from 'length' if available
                if "length" in c_dict:
                    ret_dict["km"] = c_dict["length"]
                if "waste" in c_dict:
                    ret_dict["kg"] = c_dict["waste"]
                # Initialize fields that can't be computed without simulation
                if "overflows" in c_dict:
                    ret_dict["overflows"] = c_dict["overflows"]
                if "total" in c_dict:
                    ret_dict["total"] = c_dict["total"]

            return ucost.data.cpu(), ret_dict, {}

        with torch.no_grad():
            ucost, c_dict, attn_dict = agent.compute_batch_sim(
                move_to(bat, opts["device"], non_blocking=True),
                (move_to(d_matrix, opts["device"], non_blocking=True) if d_matrix is not None else None),
            )
        return ucost, c_dict, attn_dict

    set_decode_type(model, "greedy")
    model.eval()
    if opts.get("temporal_horizon", 0) > 0 and opts.get("model") in ["tam"]:
        dataset.fill_history = torch.zeros((opts["val_size"], opts["graph_size"], opts["temporal_horizon"]))
        dataset.fill_history[:, :, -1] = torch.stack(
            [
                instance.get("waste", instance.get("noisy_waste", instance.get("real_waste")))
                for instance in dataset.data
            ]
        )

    all_costs = {"overflows": [], "kg": [], "km": []}
    all_ucosts = move_to(torch.tensor([]), opts["device"], non_blocking=True)
    attention_dict = {"attention_weights": [], "graph_masks": []}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts["eval_batch_size"],
        pin_memory=True,
        num_workers=opts.get("num_workers", 0),
    )

    # Use dist_matrix if provided, else attempt to use dataset.dist_matrix
    d_mat = dist_matrix if dist_matrix is not None else getattr(dataset, "dist_matrix", None)

    print("Validating...")
    for bat_id, bat in enumerate(tqdm(dataloader, disable=opts.get("no_progress_bar", True))):
        bat = prepare_batch(bat, bat_id, dataset, dataloader, opts)
        ucost, cost_dict, attn_dict = _eval_model_bat(bat, d_mat)

        for key in attention_dict.keys():
            if key in attn_dict:
                attention_dict[key].append(attn_dict[key])

        all_ucosts = torch.cat((all_ucosts, ucost), 0)
        for key, val in cost_dict.items():
            if key not in all_costs:
                all_costs[key] = []
            all_costs[key].append(val)

    # Concatenate results
    for key, val in attention_dict.items():
        if val:
            attention_dict[key] = torch.cat(val)

    for key, val in all_costs.items():
        if val:
            all_costs[key] = torch.cat(val)

    avg_ucost = all_ucosts.mean()

    # 1. HPO Scoring Case
    if metric is not None:
        if metric == "overflows":
            cost = all_costs[metric]
        elif metric == "kg/km":
            eps = 1e-8
            cost = all_costs["kg"] / all_costs["km"].clamp(min=eps)
        elif metric == "both":
            eps = 1e-8
            cost = all_costs["kg"] / all_costs["km"].clamp(min=eps) - all_costs["overflows"]
        else:
            cost = all_ucosts  # Default to raw cost if metric unrecognized but provided

        metric_score = cost.mean()
        print(
            "Validation overall {} score: {} +- {}".format(
                metric,
                metric_score,
                torch.std(cost) / math.sqrt(len(cost)) if len(cost) > 1 else 0.0,
            )
        )
        return metric_score, avg_ucost, all_costs

    # 2. Adaptive Weight Update Case
    if cw_dict is not None:
        eps = 1e-8
        overflows_mean = torch.mean(all_costs["overflows"].float())
        kg_mean = torch.mean(all_costs["kg"].float())
        km_mean = torch.mean(all_costs["km"].float())
        efficiency = kg_mean / km_mean.clamp(min=eps)

        kg_lost = torch.tensor(0.0)
        if "kg_lost" in all_costs:
            kg_lost = torch.mean(all_costs["kg_lost"])
        else:
            kg_lost = overflows_mean * 100 - kg_mean

        target_efficiency = 12.5
        efficiency_gap = (target_efficiency - efficiency) / target_efficiency
        overflow_gap = (overflows_mean - 372.5) / 372.5
        waste_ratio = kg_lost / kg_mean

        max_adaptation = min(opts.get("adaptation_rate", 0.1), 0.2)

        def bounded_sigmoid(x):
            """
            Map input to a stable [-1, 1] range using a shifted and scaled sigmoid.
            """
            return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0

        overflow_adjust = max_adaptation * bounded_sigmoid(overflow_gap)
        efficiency_adjust = max_adaptation * bounded_sigmoid(efficiency_gap)
        waste_adjust = -max_adaptation * bounded_sigmoid(waste_ratio * 5.0)

        damping = 0.7
        overflow_w = cw_dict["overflows"] * (1 - damping) + (cw_dict["overflows"] * (1 + overflow_adjust)) * damping
        waste_w = cw_dict["waste"] * (1 - damping) + (cw_dict["waste"] * (1 + waste_adjust)) * damping
        length_w = cw_dict["length"] * (1 - damping) + (cw_dict["length"] * (1 + efficiency_adjust)) * damping

        constraint = opts.get("constraint", 3.0)
        min_weight = 0.05 * constraint
        max_weight = 0.6 * constraint

        overflow_w = overflow_w.clamp(min_weight, max_weight).item()
        waste_w = waste_w.clamp(min_weight, max_weight).item()
        length_w = length_w.clamp(min_weight, max_weight).item()

        sum_w = overflow_w + waste_w + length_w
        constraint_factor = constraint / sum_w

        new_cw = {
            "overflows": overflow_w * constraint_factor,
            "waste": waste_w * constraint_factor,
            "length": length_w * constraint_factor,
        }

        if new_cw["waste"] > 1.5 * cw_dict["waste"]:
            new_cw["waste"] = 1.5 * cw_dict["waste"]
            remaining = constraint - new_cw["waste"]
            overflow_length_ratio = overflow_w / (overflow_w + length_w)
            new_cw["overflows"] = remaining * overflow_length_ratio
            new_cw["length"] = remaining * (1 - overflow_length_ratio)

        print("New cost function weights: ")
        for key, val in new_cw.items():
            print(f"- {key}: {val}")

        print(
            "Validation overall avg_cost: {} +- {}".format(
                avg_ucost,
                (torch.std(all_ucosts) / math.sqrt(len(all_ucosts)) if len(all_ucosts) > 1 else 0.0),
            )
        )
        for key, val in all_costs.items():
            val = val.float()
            std_val = torch.std(val) / math.sqrt(len(val)) if len(val) > 1 else 0.0
            print("- {}: {} +- {}".format(key, val.mean(), std_val))

        return new_cw, avg_ucost, all_costs

    # 3. Simple Case
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_ucost,
            (torch.std(all_ucosts) / math.sqrt(len(all_ucosts)) if len(all_ucosts) > 1 else 0.0),
        )
    )
    return avg_ucost


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clip gradient norms for all parameter groups.

    Applies gradient norm clipping to prevent exploding gradients during training.
    Returns both original and clipped gradient norms for monitoring.

    Args:
        param_groups: List of parameter groups from optimizer (optimizer.param_groups)
        max_norm: Maximum allowed gradient norm (default: math.inf for no clipping)

    Returns:
        tuple: (grad_norms, clipped_grad_norms)
            - grad_norms: List of original gradient norms (L2) per parameter group
            - clipped_grad_norms: List of clipped gradient norms per parameter group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            (max_norm if max_norm > 0 else math.inf),  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def prepare_epoch(optimizer, epoch, problem, tb_logger, cost_weights, opts):
    """
    Prepare a training epoch by initializing dataset and logging configuration.

    This function is called at the start of each training epoch to:
    1. Log the epoch/day start with current learning rate
    2. Initialize the loss tracking keys
    3. Create or load the training dataset
    4. Calculate the global step counter

    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch/day number
        problem: Problem environment (VRPP, WCVRP, etc.)
        tb_logger: TensorBoard logger
        cost_weights: Dictionary of cost function weights
        opts: Training options dictionary

    Returns:
        tuple: (step, training_dataset, loss_keys)
            - step: Global step counter for this epoch
            - training_dataset: Dataset for this epoch
            - loss_keys: List of loss component names to track
    """
    print(
        "Start train {} {}, lr={} for run {}".format(
            "day" if opts["train_time"] else "epoch",
            epoch,
            optimizer.param_groups[0]["lr"],
            opts["run_name"],
        )
    )
    step = epoch * (opts["epoch_size"] // opts["batch_size"])
    if not opts["no_tensorboard"]:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], step)

    loss_keys = list(cost_weights.keys()) + ["total", "nll", "reinforce_loss"]
    if opts["baseline"] is not None:
        loss_keys.append("baseline_loss")
    if opts.get("imitation_weight", 0) > 0:
        loss_keys.append("imitation_loss")
    if opts["train_time"] and opts["train_dataset"] is not None:
        training_dataset = problem.make_dataset(
            filename=opts["train_dataset"],
            area=opts["area"],
            waste_type=opts["waste_type"],
            distribution=opts["data_distribution"],
            vertex_strat=opts["vertex_method"],
            dist_matrix_path=opts["dm_filepath"],
            dist_strat=opts["distance_method"],
            number_edges=opts["edge_threshold"],
            edge_strat=opts["edge_method"],
            focus_graph=opts["focus_graph"],
            focus_size=opts["focus_size"],
            size=opts["graph_size"],
            num_samples=opts["epoch_size"],
        )
    elif not opts["train_time"] and opts["train_dataset"] is not None:
        filename = os.path.join(
            "data",
            "datasets",
            problem.NAME,
            "{}{}{}_{}{}_seed{}.pkl".format(
                problem.NAME,
                opts["graph_size"],
                ("_{}".format(opts["data_distribution"]) if opts["data_distribution"] is not None else ""),
                opts["train_dataset"],
                epoch,
                opts["seed"],
            ),
        )
        training_dataset = problem.make_dataset(
            filename=filename,
            number_edges=opts["edge_threshold"],
            edge_strat=opts["edge_method"],
            size=opts["graph_size"],
            dist_matrix_path=opts["dm_filepath"],
            num_samples=opts["epoch_size"],
            dist_strat=opts["distance_method"],
            area=opts["area"],
            distribution=opts["data_distribution"],
            vertex_strat=opts["vertex_method"],
            waste_type=opts["waste_type"],
            focus_graph=opts["focus_graph"],
            focus_size=opts["focus_size"],
        )
    else:
        training_dataset = problem.make_dataset(
            size=opts["graph_size"],
            num_samples=opts["epoch_size"],
            dist_strat=opts["distance_method"],
            distribution=opts["data_distribution"],
            vertex_strat=opts["vertex_method"],
            area=opts["area"],
            number_edges=opts["edge_threshold"],
            edge_strat=opts["edge_method"],
            focus_size=opts["focus_size"],
            waste_type=opts["waste_type"],
            focus_graph=opts["focus_graph"],
            dist_matrix_path=opts["dm_filepath"],
        )
    return step, training_dataset, loss_keys


def prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts):
    """
    Prepare the dataset for time-based training (e.g., WCVRP simulation).

    Initializes or updates the dataset for the current simulation day, handling
    bin selection, waste generation, and temporal horizon setup.

    Args:
        optimizer: Optimizer.
        day (int): Current simulation day.
        problem: Problem environment.
        tb_logger: Logger.
        cost_weights (dict): Cost weights.
        opts (dict): Options.

    Returns:
        tuple: (step, training_dataset, loss_keys, table_df, args)
    """
    if opts["problem"] in ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp"] and opts["data_distribution"] == "emp":
        data_dir = os.path.join(os.getcwd(), "data", "wsr_simulator")
        with open(os.path.join(data_dir, "bins_selection", opts["focus_graph"])) as js:
            idx = json.load(js)
        bins = Bins(
            opts["graph_size"],
            data_dir,
            sample_dist=opts["data_distribution"],
            area=opts["area"],
            indices=idx[0],
            waste_type=opts["waste_type"],
        )
    else:
        idx = [None]
        bins = None

    step, training_dataset, loss_keys = prepare_epoch(optimizer, day, problem, tb_logger, cost_weights, opts)
    if opts["temporal_horizon"] > 0:
        data_size = training_dataset.size
        graphs = torch.stack([torch.cat((x["depot"].unsqueeze(0), x["loc"])) for x in training_dataset])
        if opts["problem"] in [
            "vrpp",
            "cvrpp",
            "wcvrp",
            "cwcvrp",
            "sdwcvrp",
            "scwcvrp",
        ]:
            for day_id in range(1, opts["temporal_horizon"] + 1):
                training_dataset["fill{}".format(day_id)] = torch.from_numpy(
                    generate_waste_prize(
                        opts["graph_size"],
                        opts["data_distribution"],
                        graphs,
                        data_size,
                        bins,
                    )
                ).float()

            if opts["model"] in ["tam"]:
                training_dataset.fill_history = torch.zeros(
                    (opts["epoch_size"], opts["graph_size"], opts["temporal_horizon"])
                ).float()
                training_dataset.fill_history[:, :, -1] = torch.stack(
                    [
                        instance.get(
                            "waste",
                            instance.get("noisy_waste", instance.get("real_waste")),
                        )
                        for instance in training_dataset.data
                    ]
                )

    # Setup for logging
    stat_keys = ["mean", "std", "min", "max"]
    col_multi_index = pd.MultiIndex.from_product([loss_keys, stat_keys])
    # day_col = pd.MultiIndex.from_tuples([('day', '')])
    # all_columns = day_col.append(col_multi_index)
    table_df = pd.DataFrame(columns=col_multi_index)
    return step, training_dataset, loss_keys, table_df, (bins,)


def complete_train_pass(
    model,
    optimizer,
    baseline,
    lr_scheduler,
    val_dataset,
    epoch,
    step,
    epoch_duration,
    tb_logger,
    cost_weights,
    opts,
    manager=None,
):
    """
    Complete a training epoch with validation, checkpointing, and cleanup.

    This function is called at the end of each training epoch to:
    1. Log epoch completion time
    2. Save model checkpoint (if configured)
    3. Run validation on val_dataset
    4. Update baseline model
    5. Step learning rate scheduler
    6. Clear CUDA cache

    Args:
        model: Neural model being trained
        optimizer: PyTorch optimizer
        baseline: Baseline object for variance reduction
        lr_scheduler: Learning rate scheduler (or None)
        val_dataset: Validation dataset
        epoch: Current epoch/day number
        step: Global step counter
        epoch_duration: Time taken for this epoch (seconds)
        tb_logger: TensorBoard logger
        cost_weights: Dictionary of cost function weights
        opts: Training options dictionary
        manager: Meta-learning manager (optional)

    Returns:
        None
    """
    print(
        "Finished {} {}, took {} s".format(
            "day" if opts["train_time"] else "epoch",
            epoch,
            time.strftime("%H:%M:%S", time.gmtime(epoch_duration)),
        )
    )
    if (opts["checkpoint_epochs"] != 0 and epoch % opts["checkpoint_epochs"] == 0) or epoch == opts["n_epochs"] - 1:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
                "manager": manager.state_dict() if manager is not None else None,
            },
            os.path.join(opts["save_dir"], "epoch-{}.pt".format(epoch)),
        )

    if opts["val_size"] > 0:
        avg_reward = validate_update(model, val_dataset, opts)
        # new_cw, avg_cost, _ = validate_update(model, val_dataset, cost_weights, opts)
        if not opts["no_tensorboard"]:
            tb_logger.log_value("val_avg_cost", avg_reward, step)

    baseline.epoch_callback(model, epoch)
    if lr_scheduler is not None:
        # Only step scheduler if optimizer step has occurred (avoid 1.1.0+ warning)
        # Check standard PyTorch internal flag or assumption
        if not hasattr(optimizer, "_step_count") or optimizer._step_count > 0:
            lr_scheduler.step()
    if opts["device"] == "cuda":
        torch.cuda.empty_cache()
    return None


def prepare_batch(batch, batch_id, dataset, dataloader, opts, day=1):
    """
    Prepare a batch for model input by adding dataset-specific information.

    This function augments the batch dictionary with:
    1. Fill history for temporal models (TAM)
    2. Proper fill day indexing for multi-day horizons
    3. Edge information for graph-based encoders
    4. Distance matrix for routing problems

    Args:
        batch: Dictionary containing batch data from dataloader
        batch_id: Index of current batch
        dataset: The dataset being iterated
        dataloader: The dataloader producing batches
        opts: Options dictionary with model configuration
        day: Current day for time-based training (default: 1)

    Returns:
        dict: Augmented batch dictionary ready for model input
    """
    if opts.get("model") in ["tam"] and opts.get("temporal_horizon", 0) > 0:
        batch_size = dataloader.batch_size
        start_idx = batch_id * batch_size
        end_idx = min((batch_id + 1) * batch_size, len(dataset))
        batch_idx = torch.arange(start_idx, end_idx)
        batch["fill_history"] = dataset.fill_history[batch_idx]

    counter = 0
    filldays = ["fill{}".format(day_id) for day_id in range(day, day + opts.get("temporal_horizon", 0))]
    for k, v in batch.items():
        if "fill" in k:
            if k in filldays:
                counter += 1
                batch["fill{}".format(counter)] = v
        else:
            batch[k] = v
    if opts.get("focus_graph") is not None:
        if opts.get("encoder") in ["gac", "tgc"]:
            batch["edges"] = dataset.edges.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1).float()
        else:
            batch["edges"] = dataset.edges.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1).bool()
    if dataset.dist_matrix is not None:
        batch["dist"] = dataset.dist_matrix.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1)
    else:
        batch["dist"] = None
    return batch


def update_time_dataset(model, optimizer, dataset, routes, day, opts, args, costs=None):
    """
    Update dataset state for time-based training after a day's routes.

    This function simulates the passage of time in waste collection by:
    1. Emptying bins that were visited in the routes
    2. Adding new waste accumulation (daily fill)
    3. Generating future fill amounts for the temporal horizon
    4. Updating fill history for temporal models (TAM)
    5. Handling POMO augmentation if used

    The dataset is modified in-place to reflect the state for the next training day.

    Args:
        model: Neural model (used for TAM fill history updates)
        optimizer: PyTorch optimizer
        dataset: Dataset to update
        routes: Generated routes from previous day [batch_size, max_seq_len]
        day: Current day number
        opts: Options dictionary
        args: Additional arguments (e.g., bins object for empirical data)
        costs: Costs for POMO samples (optional, for selecting best route)

    Returns:
        Dataset: The updated dataset (modified in-place)
    """
    data_size = dataset.size
    if isinstance(routes, list):
        routes = torch.cat(routes, 0)

    routes = routes.contiguous().view(-1, routes.size(-1))

    # Check for POMO expansion
    if routes.size(0) > dataset.size:
        # Assuming routes are (Batch * POMO, SeqLen)
        # We need to reduce to (Batch, SeqLen) for environment update
        # Just take the first POMO sample for each instance (canonical)
        pomo_size = routes.size(0) // dataset.size
        routes = routes.view(dataset.size, pomo_size, -1)
        if costs is not None:
            if isinstance(costs, list):
                costs = torch.cat(costs, 0)
            costs = costs.view(dataset.size, pomo_size)
            best_idx = costs.argmin(dim=1)
            routes = routes[torch.arange(dataset.size), best_idx, :]
        else:
            routes = routes[:, 0, :]
    graphs = torch.stack([torch.cat((x["depot"].unsqueeze(0), x["loc"])) for x in dataset])

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    if opts["problem"] in ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"]:
        # Get masks for bins present in routes
        dataset_dim = routes.size(0)
        waste = torch.stack(
            [
                torch.cat(
                    (
                        torch.tensor([0]),
                        x.get("waste", x.get("noisy_waste", x.get("real_waste"))),
                    )
                )
                for x in dataset
            ]
        )
        num_nodes = waste.size(1)

        sorted_routes = routes.sort(1)[0]
        visited_mask = torch.zeros((dataset_dim, num_nodes), dtype=torch.bool).to(sorted_routes.device)
        col_idx = sorted_routes[sorted_routes != 0]
        row_idx = torch.arange(dataset_dim, device=sorted_routes.device).repeat_interleave(
            (sorted_routes != 0).sum(dim=1)
        )
        visited_mask[row_idx, col_idx] = True

        # Set waste in visited bins to 0 and remove waste above max_waste
        max_waste = torch.stack([x["max_waste"] for x in dataset]).unsqueeze(1)
        waste[visited_mask] = 0
        waste = waste[:, 1:].clamp_(max=max_waste)

        # Add daily waste filling
        fillday = f"fill{day + 1}"
        if opts["temporal_horizon"] > 0:
            waste += torch.stack([x[fillday] for x in dataset])
            for day_id in range(day + 1, day + opts["temporal_horizon"] + 1):
                fill_key = "fill{}".format(day_id)
                if day_id > opts["n_epochs"]:
                    dataset[fill_key] = torch.zeros((dataset_dim, waste.size(-1)), dtype=torch.float).to(
                        sorted_routes.device
                    )
                elif fill_key in dataset[0].keys():
                    dataset[fill_key] = torch.stack([x[fill_key] for x in dataset])
                else:
                    dataset[fill_key] = torch.from_numpy(
                        generate_waste_prize(
                            opts["graph_size"],
                            opts["data_distribution"],
                            graphs,
                            data_size,
                            *args,
                        )
                    ).float()
            if opts["model"] in ["tam"]:
                dataset.fill_history = get_inner_model(model).update_fill_history(dataset.fill_history, waste)
        else:
            if fillday in dataset[0].keys():
                fill = torch.stack([x[fillday] for x in dataset])
            else:
                fill = torch.from_numpy(
                    generate_waste_prize(
                        opts["graph_size"],
                        opts["data_distribution"],
                        graphs,
                        data_size,
                        *args,
                    )
                ).float()
            waste += fill
        dataset["waste"] = torch.clone(waste).to(dtype=torch.float).clamp(max=max_waste)
    else:
        raise ValueError("Problem {} not supported".format(opts["problem"]))
    is_val = data_size == opts["val_size"]
    print(
        "Start {} day {},{} for run {}".format(
            "eval" if is_val else "train",
            day,
            " lr={}".format(optimizer.param_groups[0]["lr"]) if not is_val else "",
            opts["run_name"],
        )
    )
    return dataset
