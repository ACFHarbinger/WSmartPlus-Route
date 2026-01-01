import torch


from typing import Dict, List, Union
import numpy as np
from logic.src.utils.definitions import DAY_METRICS
from logic.src.utils.functions import move_to

def set_daily_waste(model_data, waste, device, fill=None):
    model_data['waste'] = torch.as_tensor(waste, dtype=torch.float32).unsqueeze(0)/100.
    if 'fill_history' in model_data: 
        model_data['current_fill'] = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0)/100.
    return move_to(model_data, device)

def get_daily_results(total_collected, ncol, cost, tour, day, new_overflows, sum_lost, coordinates, profit):
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
    from logic.src.utils.log_utils import send_daily_output_to_gui
    return send_daily_output_to_gui(*args, **kwargs)

def run_day(graph_size, pol, bins, new_data, coords, run_tsp, sample_id, overflows, 
            day, model_env, model_ls, n_vehicles, area, realtime_log_path, waste_type, 
            distpath_tup, current_collection_day, cached, device, lock=None, hrl_manager=None,
            gate_prob_threshold=0.5, mask_prob_threshold=0.5, two_opt_max_iter=0):
    
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