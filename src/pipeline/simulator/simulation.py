import os
import time
import torch
import statistics
import numpy as np
import pandas as pd

from tqdm import tqdm
from src.utils.definitions import (
    ROOT_DIR, TQDM_COLOURS,
    SIM_METRICS, DAY_METRICS, 
)
from src.utils.setup_utils import setup_model, setup_env
from src.utils.log_utils import log_to_json, output_stats
from .bins import Bins
from .checkpoints import (
    checkpoint_manager,
    SimulationCheckpoint, 
    CheckpointHook, CheckpointError,
)
from .day import run_day
from .loader import load_depot, load_simulator_data
from .processor import process_data, process_model_data
from .network import (
    get_paths_between_states,
    compute_distance_matrix, apply_edges
)


# Create a global variable/placeholder for the lock and counter
_lock = None
_counter = None

def init_single_sim_worker(lock_from_main, counter_from_main):
    """
    Initializes globals for the single_simulation worker.
    """
    global _lock
    global _counter
    _lock = lock_from_main
    _counter = counter_from_main


def save_matrix_to_excel(matrix, results_dir, seed, data_dist, policy, sample_id):
    # Função para gerar a matriz de enchimento diária que foi usada pela politica
    parent_dir = os.path.join(results_dir, 'fill_history', data_dist)
    fills_filepath = os.path.join(parent_dir, f"enchimentos_seed{seed}_sample{sample_id}.xlsx")
    if os.path.exists(fills_filepath) and os.path.isfile(fills_filepath):
        return
    
    df = pd.DataFrame(matrix).transpose()
    filepath = os.path.join(parent_dir, f"{policy}{seed}_sample{sample_id}.xlsx")
    df.to_excel(filepath, index=False, header=False)
    return


def _setup_basedata(n_bins, data_dir, area, waste_type):
    depot = load_depot(data_dir, area)
    data, bins_coordinates = load_simulator_data(data_dir, n_bins, area, waste_type)
    assert data.shape == bins_coordinates.shape
    return data, bins_coordinates, depot


def _setup_dist_path_tup(bins_coordinates, size, dist_method, dm_filepath, env_filename, 
                        gapik_file, symkey_name, device, edge_thresh, edge_method, focus_idx=None):
    dist_matrix = compute_distance_matrix(bins_coordinates, dist_method, dm_filepath=dm_filepath, env_filename=env_filename, 
                                          gapik_file=gapik_file, symkey_name=symkey_name, focus_idx=focus_idx)
    dist_matrix_edges, shortest_paths, adj_matrix = apply_edges(dist_matrix, edge_thresh, edge_method)
    paths = get_paths_between_states(size+1, shortest_paths)
    dm_tensor = torch.from_numpy(dist_matrix_edges).to(device)
    distC = np.round(dist_matrix_edges*10).astype('int32')
    return (dist_matrix_edges, paths, dm_tensor, distC), adj_matrix


def display_log_metrics(output_dir, size, n_samples, days, area, policies, log, log_std=None, lock=None):
    if n_samples > 1:
        output_dir = os.path.join(ROOT_DIR, "assets", output_dir, str(days) + "_days", str(area) + '_' + str(size))
        dit = {}
        std_dit = {}
        for pol, lg, lg_st in zip(policies, log, log_std):
            dit[pol] = lg
            std_dit[pol] = lg_st
        log_to_json(os.path.join(output_dir, f"log_mean_{n_samples}N.json"), SIM_METRICS, log, sample_id=None, lock=lock)
        log_to_json(os.path.join(output_dir, f"log_std_{n_samples}N.json"), SIM_METRICS, log_std, sample_id=None, lock=lock)
        for lg, lg_std, pol in zip(log.values(), log_std.values(), log.keys()):
            logm = lg.values() if isinstance(lg, dict) else lg
            logs = lg_std.values() if isinstance(lg_std, dict) else lg_std
            tmp_lg = [(str(x), str(y)) for x, y in zip(logm, logs)]
            print(f"\n{pol} log:")
            for (x, y), key in zip(tmp_lg, SIM_METRICS):
                print(f"- {key} value: {x[:x.find('.')+3]} +- {y[:y.find('.')+5]}")
    else:
        for pol, lg in log.items():
            print(f"\n{pol} log:")
            for key, val in zip(SIM_METRICS, lg):
                print(f"- {key}: {val}")
    return


def single_simulation(opts, device, indices, sample_id, pol_id, model_weights_path, n_cores):
    def _get_current_state():
        return (
            new_data, coords, dist_tup, adj_matrix, 
            bins, model_tup, cached, overflows, 
            current_collection_day, daily_log, execution_time
        )
    # Retrieve the shared objects via the global variables initialized by init_worker
    global _lock
    global _counter

    checkpoint = None
    data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
    results_dir = os.path.join(ROOT_DIR, "assets", opts['output_dir'], str(opts['days']) + "_days", str(opts['area']) + '_' + str(opts['size']))
    
    policy = opts['policies'][pol_id]
    pol_strip, data_dist = policy.rsplit("_", 1)
    data, bins_coordinates, depot = _setup_basedata(opts['size'], data_dir, opts['area'], opts['waste_type'])
    checkpoint = SimulationCheckpoint(results_dir, opts['checkpoint_dir'], policy, sample_id)

    # Try to load previous state
    saved_state, last_day = (None, 0)
    if opts['resume']: 
        saved_state, last_day = checkpoint.load_state()

    if 'am' in pol_strip or "transgcn" in pol_strip:
        model_env, configs = setup_model(policy, model_weights_path, device, _lock, opts['temperature'], opts['decode_type'])
    elif "vrpp" in pol_strip:
        model_env = setup_env(policy, opts['server_run'], opts['gplic_file'], opts['symkey_name'], opts['env_file'])
        model_tup = (None, None)
        configs = None
    else:
        configs = None
        model_env = None
        model_tup = (None, None)

    if opts['resume'] and saved_state is not None:
        (new_data, coords, dist_tup, adj_matrix, 
        bins, model_tup, cached, overflows, 
        current_collection_day, daily_log, run_time) = saved_state
        start_day = last_day + 1
    else:
        new_data, coords = process_data(data, bins_coordinates, depot, indices)
        dist_tup, adj_matrix = _setup_dist_path_tup(coords, opts['size'], opts['distance_method'], opts['dm_filepath'], 
                                                    opts['env_file'], opts['gapik_file'], opts['symkey_name'], device, 
                                                    opts['edge_threshold'], opts['edge_method'], indices)
        if 'am' in pol_strip or "transgcn" in pol_strip:
            model_tup = process_model_data(coords, dist_tup[-1], device, opts['vertex_method'], 
                                        configs, opts['edge_threshold'], opts['edge_method'], 
                                        opts['area'], opts['waste_type'], adj_matrix)
        if "gamma" in data_dist:
            bins = Bins(opts['size'], data_dir, data_dist[:-1], area=opts['area'], waste_file=opts['waste_filepath'])
            gamma_option = int(policy[-1]) - 1
            bins.setGammaDistribution(option=gamma_option)
        else:
            assert data_dist == "emp"
            bins = Bins(opts['size'], data_dir, data_dist, area=opts['area'], waste_file=opts['waste_filepath'])

        cached = [] if opts['cache_regular'] else None
        if opts['waste_filepath'] is not None:
            bins.set_sample_waste(sample_id)

        run_time = 0
        overflows = 0
        start_day = 1
        current_collection_day = 0
        bins.set_indices(indices)
        daily_log = {key: [] for key in DAY_METRICS}
    
    attention_dict = {}
    desc = f"{policy} #{sample_id}"
    colour = TQDM_COLOURS[pol_id % len(TQDM_COLOURS)]
    tqdm_position = os.getpid() % n_cores + 1
    log_path = os.path.join(results_dir, f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.json")
    tic = time.process_time() + run_time
    try:
        with checkpoint_manager(checkpoint, opts['checkpoint_days'], _get_current_state) as hook:
            hook.set_timer(tic)
            for day in tqdm(range(start_day, opts['days']+1), disable=opts['no_progress_bar'], desc=desc, 
                            position=tqdm_position+1, dynamic_ncols=True, leave=False, colour=colour):
                hook.before_day(day)
                data_ls, output_ls, cached = run_day(opts['size'], policy, bins, new_data, coords, opts['run_tsp'], sample_id,
                                                    overflows, day, model_env, model_tup, opts['n_vehicles'], opts['area'], 
                                                    log_path, opts['waste_type'], dist_tup, current_collection_day, cached, device)
                execution_time = time.process_time() - tic
                new_data, coords, bins = data_ls
                overflows, dlog, output_dict = output_ls
                with _counter.get_lock(): _counter.value += 1
                if 'am' in pol_strip or "transgcn" in pol_strip:
                    if pol_strip not in attention_dict:
                        attention_dict[pol_strip] = []
                    attention_dict[pol_strip].append(output_dict)
                for key, val in dlog.items():
                    daily_log[key].append(val)
                hook.after_day(execution_time)
            execution_time = time.process_time() - tic
            lg = [np.sum(bins.inoverflow), np.sum(bins.collected), np.sum(bins.ncollections), 
            np.sum(bins.lost), bins.travel, np.nan_to_num(np.sum(bins.collected)/bins.travel, 0), 
            np.sum(bins.inoverflow)-np.sum(bins.collected)+bins.travel, bins.ndays, execution_time]
            daily_log_path = os.path.join(results_dir, f"daily_{opts['data_distribution']}_{opts['n_samples']}N.json")
            if opts['n_samples'] > 1:
                log_path = os.path.join(results_dir, f"log_full_{opts['n_samples']}N.json")
                log_to_json(log_path, SIM_METRICS, {policy: lg}, sample_id=sample_id, lock=_lock)
                log_to_json(daily_log_path, DAY_METRICS, {f"{pol_strip} #{sample_id}": daily_log.values()}, lock=_lock)
            else:
                log_path = os.path.join(results_dir, f"log_mean_{opts['n_samples']}N.json")
                log_to_json(log_path, SIM_METRICS, {policy: lg}, lock=_lock)
                log_to_json(daily_log_path, DAY_METRICS, {pol_strip: daily_log.values()}, lock=_lock)
            
            # Save fill history and clear checkpoints after successful completion
            save_matrix_to_excel(bins.get_fill_history(), results_dir, opts['seed'], 
                                opts['data_distribution'], policy, sample_id)
            hook.on_completion(policy, sample_id)
            return {policy: lg, 'success': True}
    except CheckpointError as e:
        return e.error_result
    except Exception as e:
        raise


def sequential_simulations(opts, device, indices_ls, sample_idx_ls, model_weights_path, lock):
    def _get_current_state():
        return (
            new_data, coords, dist_tup, adj_matrix, 
            bins, model_tup, cached, overflows, 
            current_collection_day, daily_log, execution_time
        )

    log = {}
    failed_log = []
    if opts['n_samples'] > 1:
        log_std = {}
        log_full = dict.fromkeys(opts['policies'], [])
    else:
        log_std = None

    # Create overall progress bar FIRST with position=1
    overall_progress = tqdm(total=sum(len(sublist) for sublist in sample_idx_ls) * opts['days'], 
                            desc="Overall progress", disable=opts['no_progress_bar'], 
                            position=1, leave=True)  # Ensure it stays visible

    data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
    results_dir = os.path.join(ROOT_DIR, "assets", opts['output_dir'], 
                               str(opts['days']) + "_days", 
                               str(opts['area']) + '_' + str(opts['size']))
    daily_log_path = os.path.join(results_dir, f"daily_{opts['data_distribution']}_{opts['n_samples']}N.json")
    log_path = os.path.join(results_dir, f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl")
    data, bins_coordinates, depot = _setup_basedata(opts['size'], data_dir, opts['area'], opts['waste_type'])
    for pol_id, policy in enumerate(opts['policies']):
        pol_strip, data_dist = policy.rsplit("_", 1)
        if 'am' in pol_strip or "transgcn" in pol_strip:
            attention_dict = {pol_strip: []}
            model_env, configs = setup_model(policy, model_weights_path, device, lock, 
                                             opts['temperature'], opts['decode_type'])
        elif "vrpp" in pol_strip:
            model_env = setup_env(policy, opts['server_run'], opts['gplic_file'], 
                                  opts['symkey_name'], opts['env_file'])
            configs = {"problem": opts['problem']}
        else:
            model_env = None
            configs = {"problem": opts['problem']}

        for sample_id in sample_idx_ls[pol_id]:
            try:
                # Initialize variables to avoid reference errors
                last_day = 0
                saved_state = None
                checkpoint = SimulationCheckpoint(results_dir, opts['checkpoint_dir'], policy, sample_id)
                if opts['resume']:
                    saved_state, last_day = checkpoint.load_state()

                desc = f"{policy} #{sample_id}"
                if opts['resume'] and saved_state is not None:
                    (new_data, coords, dist_tup, adj_matrix, 
                    bins, model_tup, cached, overflows, 
                    current_collection_day, daily_log, run_time) = saved_state
                    start_day = last_day + 1

                    # Update overall progress for already completed days
                    completed_days = last_day
                    overall_progress.update(completed_days)
                else:
                    new_data, coords = process_data(data, bins_coordinates, depot, indices_ls[sample_id])
                    dist_tup, adj_matrix = _setup_dist_path_tup(coords, opts['size'], opts['distance_method'], opts['dm_filepath'], 
                                                                opts['env_file'], opts['gapik_file'], opts['symkey_name'], device,
                                                                opts['edge_threshold'], opts['edge_method'], indices_ls[sample_id])
                    if 'am' in pol_strip or "transgcn" in pol_strip:
                        model_tup = process_model_data(coords, dist_tup[-1], device, opts['vertex_method'], 
                                                       configs, opts['edge_threshold'], opts['edge_method'], 
                                                       opts['area'], opts['waste_type'], adj_matrix)
                    else:
                        model_tup = (None, None)
                    if "gamma" in data_dist:
                        bins = Bins(opts['size'], data_dir, data_dist[:-1], area=opts['area'], waste_file=opts['waste_filepath'])
                        gamma_option = int(policy[-1]) - 1
                        bins.setGammaDistribution(option=gamma_option)
                    else:
                        assert data_dist == "emp"
                        bins = Bins(opts['size'], data_dir, data_dist, area=opts['area'], waste_file=opts['waste_filepath'])

                    cached = [] if opts['cache_regular'] else None
                    if opts['waste_filepath'] is not None:
                        bins.set_sample_waste(sample_id)

                    run_time = 0
                    start_day = 1
                    overflows = 0
                    current_collection_day = 0
                    bins.set_indices(indices_ls[sample_id])
                    daily_log = {key: [] for key in DAY_METRICS}
        
                hook = CheckpointHook(checkpoint, opts['checkpoint_days'], _get_current_state)
                colour = TQDM_COLOURS[pol_id % len(TQDM_COLOURS)]
                
                # Individual policy progress bar with position=2 (BELOW overall)
                policy_progress = tqdm(opts['days'], initial=start_day-1, desc=desc,
                                      colour=colour, position=2, leave=False,
                                      disable=opts['no_progress_bar'])
                
                tic = time.process_time() + run_time
                hook.set_timer(tic)
                for day in range(start_day, opts['days']+1):
                    hook.before_day(day)
                    data_ls, output_ls, cached = run_day(opts['size'], policy, bins, new_data, coords, opts['run_tsp'], sample_id, 
                                                        overflows, day, model_env, model_tup, opts['n_vehicles'], opts['area'], 
                                                        log_path, opts['waste_type'], dist_tup, current_collection_day, cached, device)
                    execution_time = time.process_time() - tic
                    new_data, coords, bins = data_ls
                    overflows, dlog, output_dict = output_ls
                    if 'am' in pol_strip or "transgcn" in pol_strip: 
                        attention_dict[pol_strip].append(output_dict)
                    for key, val in dlog.items():
                        daily_log[key].append(val)
                    hook.after_day(execution_time)
                    
                    # Update both progress bars
                    policy_progress.update(1)
                    overall_progress.update(1)
                    policy_progress.refresh()
                    overall_progress.refresh()
                
                # Close individual policy progress bar
                policy_progress.close()
                
                execution_time = time.process_time() - tic
                lg = [np.sum(bins.inoverflow), np.sum(bins.collected), np.sum(bins.ncollections), 
                np.sum(bins.lost), bins.travel, np.nan_to_num(np.sum(bins.collected)/bins.travel, 0), 
                np.sum(bins.inoverflow)-np.sum(bins.collected)+bins.travel, bins.ndays, execution_time]
                if opts['n_samples'] > 1:
                    log_full[policy].append(lg)
                    log_path = os.path.join(results_dir, f"log_full_{opts['n_samples']}N.json")
                    log_to_json(log_path, SIM_METRICS, {policy: lg}, sample_id=sample_id, lock=lock)
                    log_to_json(daily_log_path, DAY_METRICS, {f"{pol_strip} #{sample_id}": daily_log.values()}, lock=lock)
                else:
                    log[policy] = lg
                    log_path = os.path.join(results_dir, f"log_mean_{opts['n_samples']}N.json")
                    log_to_json(log_path, SIM_METRICS, {policy: lg}, lock=lock)
                    log_to_json(daily_log_path, DAY_METRICS, {pol_strip: daily_log.values()}, lock=lock)
                
                # Save fill history and clear checkpoints after successful completion
                save_matrix_to_excel(bins.get_fill_history(), results_dir, opts['seed'], 
                                    opts['data_distribution'], policy, sample_id)
                hook.on_completion(policy, sample_id)
                
            except CheckpointError as e:
                failed_log.append(e.error_result)
                if 'start_day' in locals() and 'day' in locals():
                    completed_in_this_run = day - start_day
                    overall_progress.update(completed_in_this_run)
                continue
            except Exception as e:
                if 'start_day' in locals() and 'day' in locals():
                    completed_in_this_run = day - start_day + 1
                    overall_progress.update(completed_in_this_run)
                raise e

        if opts['n_samples'] > 1:
            if opts['resume']:
                log, log_std = output_stats(ROOT_DIR, opts['days'], opts['size'], opts['output_dir'], 
                                            opts['area'], opts['n_samples'], [policy], SIM_METRICS, lock)
            else:                
                log[policy] = [*map(statistics.mean, zip(*log_full[policy]))]
                log_std[policy] = [*map(statistics.stdev, zip(*log_full[policy]))]
                log_to_json(os.path.join(results_dir, f"log_mean_{opts['n_samples']}N.json"), SIM_METRICS, {policy: log[policy]}, lock=lock)
                log_to_json(os.path.join(results_dir, f"log_std_{opts['n_samples']}N.json"), SIM_METRICS, {policy: log_std[policy]}, lock=lock)
    
    # Close overall progress bar
    overall_progress.close()
    return log, log_std, failed_log
