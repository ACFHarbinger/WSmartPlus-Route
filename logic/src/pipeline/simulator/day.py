"""
Single-Day Simulation Orchestration.

This module coordinates the execution of a single simulation day using
the Command Pattern. It prepares data, executes the action sequence
(Fill → Policy → Collect → Log), and returns structured results.

The day execution follows a fixed pipeline:
    1. Prepare context with all required inputs
    2. Execute FillAction (bin waste deposition)
    3. Execute PolicyExecutionAction (compute tour)
    4. Execute CollectAction (empty bins)
    5. Execute LogAction (record metrics)

Functions:
    set_daily_waste: Updates model input tensors with current waste levels
    get_daily_results: Formats raw outputs into structured metric dictionary
    run_day: Main orchestrator for single-day simulation
"""

import torch


from typing import Dict, List, Union
import numpy as np
from logic.src.utils.definitions import DAY_METRICS
from logic.src.utils.functions import move_to


def set_daily_waste(model_data, waste, device, fill=None):
    """
    Updates neural model input with current bin waste levels.

    Converts numpy waste arrays to PyTorch tensors and normalizes to [0, 1].
    Handles both standard and temporal models (with fill history).

    Args:
        model_data: Dictionary of model inputs (modified in-place)
        waste: Current bin fill levels (0-100%) as numpy array
        device: torch.device for tensor placement
        fill: Optional absolute fill levels for temporal models

    Returns:
        Updated model_data dict with tensors on specified device
    """
    model_data['waste'] = torch.as_tensor(waste, dtype=torch.float32).unsqueeze(0)/100.
    if 'fill_history' in model_data:
        model_data['current_fill'] = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0)/100.
    return move_to(model_data, device)


def get_daily_results(total_collected, ncol, cost, tour, day, new_overflows, sum_lost, coordinates, profit):
    """
    Formats raw simulation outputs into structured daily log dictionary.

    Computes derived metrics (kg/km efficiency, total cost) and maps
    internal bin indices to real-world bin IDs for tour visualization.

    Args:
        total_collected: Waste collected today (kg)
        ncol: Number of bins collected
        cost: Tour distance (km)
        tour: List of internal bin indices (0-based, includes depot)
        day: Current simulation day
        new_overflows: Bins that overflowed today
        sum_lost: Waste lost to overflows (kg)
        coordinates: DataFrame with bin ID mapping
        profit: Net profit (revenue - costs)

    Returns:
        Dictionary with keys from DAY_METRICS:
            - day, overflows, kg_lost, kg, ncol, km, kg/km, cost, profit, tour
    """
    dlog: Dict[str, Union[int, float, List[int]]] = {key: 0 for key in DAY_METRICS}
    dlog['day'] = day
    dlog['overflows'] = new_overflows
    dlog['kg_lost'] = sum_lost
    if len(tour) > 2 and cost > 0:
        rl_cost = new_overflows - total_collected + cost
        dlog['kg'] = total_collected
        dlog['ncol'] = ncol
        dlog['km'] = cost
        dlog['kg/km'] = total_collected / cost
        dlog['cost'] = rl_cost
        dlog['profit'] = profit
        ids = np.array([x for x in tour if x != 0])
        dlog['tour'] = [0] + coordinates.loc[ids, 'ID'].tolist() + [0]
    else:
        dlog['kg'] = 0
        dlog['ncol'] = 0
        dlog['km'] = 0
        dlog['kg/km'] = 0
        dlog['cost'] = new_overflows
        dlog['profit'] = 0
        dlog['tour'] = [0]
    return dlog

def send_daily_output_to_gui(*args, **kwargs):
    """
    Proxy function to send daily simulation updates to the GUI.
    
    This function lazily imports the utility from log_utils to avoid
    circular dependencies and forwards all arguments.
    """
    from logic.src.utils.log_utils import send_daily_output_to_gui
    return send_daily_output_to_gui(*args, **kwargs)

def run_day(graph_size, pol, bins, new_data, coords, run_tsp, sample_id, overflows,
            day, model_env, model_ls, n_vehicles, area, realtime_log_path, waste_type,
            distpath_tup, current_collection_day, cached, device, lock=None, hrl_manager=None,
            gate_prob_threshold=0.5, mask_prob_threshold=0.5, two_opt_max_iter=0, config=None):
    """
    Orchestrates a single simulation day using the Command Pattern.

    Executes the four-stage pipeline: Fill → Policy → Collect → Log.
    All state and parameters are passed via a shared context dictionary.

    Args:
        graph_size: Number of bins in the problem
        pol: Policy identifier string
        bins: Bins state manager
        new_data: Bin data DataFrame
        coords: Bin coordinate DataFrame
        run_tsp: Whether to run TSP post-optimization
        sample_id: Sample/seed identifier
        overflows: Cumulative overflow count
        day: Current simulation day
        model_env: Loaded neural model or solver environment
        model_ls: Model configuration tuple
        n_vehicles: Number of available vehicles
        area: Geographic area name
        realtime_log_path: Path for real-time GUI logging
        waste_type: Waste stream type
        distpath_tup: (distance_matrix, paths, dm_tensor, distancesC)
        current_collection_day: Day counter for periodic policies
        cached: Cache for regular policy
        device: torch.device for neural models
        lock: Thread lock for parallel simulations
        hrl_manager: Hierarchical RL manager (optional)
        gate_prob_threshold: HRL gating probability threshold
        mask_prob_threshold: HRL masking probability threshold
        two_opt_max_iter: 2-opt local search iterations
        config: Policy-specific configuration dict

    Returns:
        Tuple containing:
            - data_ls: (new_data, coords, bins) updated state
            - output_ls: (overflows, daily_log, output_dict) results
            - cached: Updated policy cache
    """

    # Prepare context
    distance_matrix, paths_between_states, dm_tensor, distancesC = distpath_tup
    
    context = {
        'policy': pol.rsplit('_', 1)[0], # Stripped policy name
        'full_policy': pol, # Original string including modifiers
        'policy_name': pol.rsplit('_', 1)[0], # Base name for factory
        'bins': bins,
        'distpath_tup': distpath_tup,
        'distance_matrix': distance_matrix,
        'distancesC': distancesC,
        'paths_between_states': paths_between_states,
        'dm_tensor': dm_tensor,
        'new_data': new_data,
        'coords': coords,
        'run_tsp': run_tsp,
        'waste_type': waste_type,
        'area': area,
        'n_vehicles': n_vehicles,
        'model_env': model_env,
        'model_ls': model_ls,
        'day': day,
        'cached': cached,
        'device': device,
        'lock': lock,
        'hrl_manager': hrl_manager,
        'gate_prob_threshold': gate_prob_threshold,
        'mask_prob_threshold': mask_prob_threshold,
        'two_opt_max_iter': two_opt_max_iter,
        'current_collection_day': current_collection_day,
        'sample_id': sample_id,
        'realtime_log_path': realtime_log_path,
        'overflows': overflows,
        'graph_size': graph_size,
        'config': config if config is not None else {}, # Pass config
        
        # Outputs placeholders
        'output_dict': None,
        'cached': cached # Will be updated if modified
    }
    
    from logic.src.pipeline.simulator.actions import (
        FillAction, PolicyExecutionAction, CollectAction, LogAction
    )
    
    commands = [
        FillAction(),
        PolicyExecutionAction(),
        CollectAction(),
        LogAction()
    ]
    
    for command in commands:
        command.execute(context)
        
    # Extract results
    return (new_data, coords, bins), (context['overflows'], context['daily_log'], context.get('output_dict')), context['cached']