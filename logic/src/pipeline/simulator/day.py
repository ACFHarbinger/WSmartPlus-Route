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
from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np
import pandas as pd
from logic.src.utils.definitions import DAY_METRICS
from logic.src.utils.functions import move_to
from logic.src.pipeline.simulator.context import SimulationDayContext


def set_daily_waste(model_data: Dict[str, Any], waste: np.ndarray, device: torch.device, fill: Optional[np.ndarray] = None) -> Dict[str, Any]:
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


def get_daily_results(total_collected: float, ncol: int, cost: float, tour: List[int], day: int, 
                      new_overflows: int, sum_lost: float, coordinates: pd.DataFrame, profit: float) -> Dict[str, Union[int, float, List[int]]]:
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
    if len(tour) > 2:
        rl_cost = new_overflows - total_collected + cost
        dlog['kg'] = total_collected
        dlog['ncol'] = ncol
        dlog['km'] = cost
        dlog['kg/km'] = total_collected / cost if cost > 0 else 0
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

def run_day(context: SimulationDayContext) -> SimulationDayContext:
    """
    Orchestrates a single simulation day using the Command Pattern.

    Executes the four-stage pipeline: Fill → Policy → Collect → Log.
    All state and parameters are passed via the shared SimulationDayContext.

    Args:
        context: SimulationDayContext object containing all simulation state.

    Returns:
        Updated SimulationDayContext.
    """
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
        
    return context