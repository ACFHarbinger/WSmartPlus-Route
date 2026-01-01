import os
import statistics

from tqdm import tqdm
from logic.src.utils.definitions import ROOT_DIR, SIM_METRICS
from logic.src.utils.log_utils import log_to_json, output_stats
from .checkpoints import CheckpointError
from .states import SimulationContext


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
    # Retrieve the shared objects via the global variables initialized by init_worker
    global _lock
    global _counter
    
    variables_dict = {
        'lock': _lock,
        'counter': _counter,
        'tqdm_pos': os.getpid() % n_cores
    }
    
    context = SimulationContext(opts, device, indices, sample_id, pol_id, model_weights_path, variables_dict)
    return context.run()


def sequential_simulations(opts, device, indices_ls, sample_idx_ls, model_weights_path, lock):
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

    # data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator") # Not needed here
    results_dir = os.path.join(ROOT_DIR, "assets", opts['output_dir'], 
                               str(opts['days']) + "_days", 
                               str(opts['area']) + '_' + str(opts['size']))
    
    for pol_id, policy in enumerate(opts['policies']):
        for sample_id in sample_idx_ls[pol_id]:
            try:
                 variables_dict = {
                    'lock': lock,
                    'overall_progress': overall_progress,
                    'tqdm_pos': 1 
                }
                 
                 context = SimulationContext(opts, device, indices_ls[sample_id], sample_id, pol_id, model_weights_path, variables_dict)
                 result_dict = context.run()
                 
                 # Aggregate execution result
                 if result_dict and 'success' in result_dict and result_dict['success']:
                     lg = result_dict[policy]
                     
                     if opts['n_samples'] > 1:
                        log_full[policy].append(lg)
                     else:
                        log[policy] = lg
                        
            except CheckpointError as e:
                failed_log.append(e.error_result)
                pass
                
            except Exception as e:
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
