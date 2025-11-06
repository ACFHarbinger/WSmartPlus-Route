import os
import sys
import json
import math
import wandb
import torch
import pickle
import datetime
import traceback
import statistics
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import backend.src.utils.definitions as udef

from collections import Counter
from collections.abc import Iterable
from backend.src.utils.io_utils import read_json


def log_values(cost, grad_norms, epoch, batch_id, step, l_dict, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('{}: {}, train_batch_id: {}, avg_cost: {}'.format(
        "day" if opts['train_time'] else "epoch", epoch, batch_id, avg_cost))
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts['no_tensorboard']:
        tb_logger.log_value('avg_cost', avg_cost, step)
        tb_logger.log_value('actor_loss', l_dict['reinforce_loss'].mean().item(), step)
        tb_logger.log_value('nll', l_dict['nll'].mean().item(), step)
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)
        if opts['baseline'] == 'critic':
            tb_logger.log_value('critic_loss', l_dict['baseline_loss'].item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)

    """if opts['wandb_mode'] != 'disabled:
        wandb.log({'avg_cost': avg_cost, 'actor_loss': l_dict['reinforce_loss'].mean().item(), 
            'nll': l_dict['nll'].mean().item(), 'grad_norm': grad_norms[0], 'grad_norm_clipped': grad_norms_clipped[0]})
        if opts['baseline'] == 'critic':
            wandb.log({'critic_loss': l_dict['baseline_loss'].item(), 'critic_grad_norm': grad_norms[1], 
                'critic_grad_norm_clipped': grad_norms_clipped[1]})"""
    return


def log_epoch(x_tup, loss_keys, epoch_loss, opts):
    for id, key in enumerate(loss_keys):
        lname = key if key in udef.LOSS_KEYS else f"{key}_cost"
        lmean = torch.cat(epoch_loss[key]).float().mean().item()
        if opts['wandb_mode'] != 'disabled':
            wandb.log({x_tup[0]: x_tup[1], lname: lmean}, commit=(key == loss_keys[-1]))
    return


def get_loss_stats(epoch_loss):
    loss_stats = []
    for key in epoch_loss.keys():
        loss_tensor = torch.cat(epoch_loss[key]).float()
        loss_tmp = [torch.mean(loss_tensor).item(), torch.std(loss_tensor).item(), 
                    torch.min(loss_tensor).item(), torch.max(loss_tensor).item()]
        loss_stats.extend(loss_tmp)
    return loss_stats


def log_training(loss_keys, table_df, opts):
    xname = "day" if opts['train_time'] else "epoch"
    x_values = [row_id for row_id in range(table_df.shape[0])]
    log_dir = os.path.join(os.getcwd(), opts['log_dir'], os.path.relpath(opts['save_dir'], start=opts['output_dir']))
    table_df.to_parquet(os.path.join(log_dir, "table.parquet"), engine="pyarrow")
    swapped_df = table_df.swaplevel(axis=1)
    swapped_df.columns = ['_'.join(col).strip() for col in swapped_df.columns]
    if opts['wandb_mode'] != 'disabled':
        wandb_table = wandb.Table(dataframe=swapped_df)
        wandb.log({'training_table': wandb_table})

    for l_id, l_key in enumerate(loss_keys):
        #mean_loss = np.array([row[wandb_table.columns.index(f"mean_{l_key}")] for row in wandb_table.data])
        #std_loss = np.array([row[wandb_table.columns.index(f"std_{l_key}")] for row in wandb_table.data])
        mean_loss = swapped_df[f"mean_{l_key}"].to_numpy()
        std_loss = swapped_df[f"std_{l_key}"].to_numpy()
        if np.all(mean_loss == std_loss): continue
        #max_loss = np.array([row[wandb_table.columns.index(f"max_{l_key}")] for row in wandb_table.data])
        #min_loss = np.array([row[wandb_table.columns.index(f"min_{l_key}")] for row in wandb_table.data])
        max_loss = swapped_df[f"max_{l_key}"].to_numpy()
        min_loss = swapped_df[f"min_{l_key}"].to_numpy()
        lower_bound = np.maximum(mean_loss - std_loss, min_loss)
        upper_bound = np.minimum(mean_loss + std_loss, max_loss)

        fig = plt.figure(l_id, figsize=(20, 10), facecolor='white', edgecolor='black', layout='constrained')
        label = l_key if l_key in udef.LOSS_KEYS else f"{l_key}_cost"
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_values, mean_loss, label=F"{label} μ", linewidth=2)
        ax.fill_between(x_values, lower_bound, upper_bound, alpha=0.3, label=f"{label} clip(μ ± σ)")
        ax.scatter(x_values, mean_loss, color='red', s=50, zorder=5)
        ax.set_xlabel(xname)
        ax.set_ylabel(label)
        ax.set_title(f"{label} per {xname}")
        ax.legend()
        ax.grid(True, linestyle='-', alpha=0.9)
        fig_path = os.path.join(log_dir, f"{label}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        if opts['wandb_mode'] != 'disabled': wandb.log({label: wandb.Image(fig_path)})


def _sort_log(log):
    log = {key: value for key, value in sorted(log.items())}
    tmp_log = {}
    for key in log.keys():
        if 'policy_last_minute' in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if 'policy_regular' in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if 'policy_look_ahead' in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if 'gurobi' in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if 'hexaly' in key:
            tmp_log[key] = log[key]
    for key in tmp_log.keys():
        log[key] = log.pop(key)
    return log


def sort_log(logfile_path, lock=None):
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        log = read_json(logfile_path, lock=None)
        log = _sort_log(log)
        with open(logfile_path, 'w') as fp:
            json.dump(log, fp, indent=True)
    finally:
        if lock is not None: lock.release()
    return


def _convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def log_to_json(json_path, keys, dit, sort_log=True, sample_id=None, lock=None):
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        read_failed = False
        if os.path.isfile(json_path):
            # IMPORTANT: Pass lock=None to read_json to avoid double-locking, 
            # since we've already acquired the lock here.
            try:
                old = read_json(json_path, lock=None) 
            except json.JSONDecodeError:
                # Handle case where file is corrupt from a previous failed write.
                # If you can't read it, assume it's empty to prevent cascading errors.
                read_failed = True
                old = [] if 'full' in json_path else {}
            
            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            if sample_id is not None:
                old = []

        for key, val in dit.items():
            values = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))

        if sort_log: new = _sort_log(new)
        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict):
                old[sample_id] = new 
        else:
            old = new

        write_error = None
        try:
            with open(json_path, 'w') as fp:
                json.dump(_convert_numpy(old), fp, indent=True)
        except (TypeError, ValueError, FileNotFoundError) as e:
            write_error = e

        if read_failed or write_error is not None:
            # Handle error on write: Send output to a temporary file.
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            filename, file_ext = os.path.splitext(json_path)
            tmp_path = filename + timestamp + "_TMP" + file_ext
            with open(tmp_path, 'w') as fp_temp:
                json.dump(_convert_numpy(old), fp_temp, indent=True)

            if read_failed and write_error is not None:
                print(f"\n[WARNING] Failed to read from and write to {json_path}.\n- Write Error: {e}")
            elif read_failed:
                print(f"\n[WARNING] Failed to read from {json_path}.")
            else:
                assert write_error is not None
                print(f"\n[WARNING] Failed to write to {json_path}.\n- Write Error: {e}")

            print(f"Data saved to temporary file: {tmp_path}")
    finally:
        if lock is not None: lock.release()  
    return old


def log_to_json2(json_path, keys, dit, sort_log=True, sample_id=None, lock=None):
    # 1. ACQUIRE LOCK (Protects the entire RMW cycle)
    if lock is not None: 
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    
    data_to_write = None # Variable to hold the final data structure to be written
    try:
        # --- READ AND MERGE DATA ---
        if os.path.isfile(json_path):
            try:
                # IMPORTANT: Pass lock=None as lock is already acquired here.
                old = read_json(json_path, lock=None) 
            except json.JSONDecodeError:
                # ⚠️ 1. Handle JSONDecodeError on read: Log the ERROR, but ASSUME empty data 
                # to continue processing the current thread's results.
                print(f"\n[WARNING] JSONDecodeError reading: {json_path}. Assuming empty data for merge.")
                old = [] if 'full' in json_path else {}
            
            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            if sample_id is not None:
                old = []

        # --- MODIFY / MERGE NEW DATA ---
        for key, val in dit.items():
            values = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))

        if sort_log: new = _sort_log(new)
        
        # Consolidate 'new' data back into 'old' structure
        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict):
                old[sample_id] = new 
        else:
            old = new
            
        data_to_write = old # Store the final merged data

        # --- WRITE DATA (With Fallback) ---
        try:
            with open(json_path, 'w') as fp:
                json.dump(_convert_numpy(data_to_write), fp, indent=True)
                
        except (TypeError, ValueError) as e:
            # ⚠️ 2. Handle ANY error on write: Send output to a temporary file.
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            temp_json_path = json_path + timestamp + "_TEMP.json"
            
            # Write to temporary file (still inside the lock for safety)
            with open(temp_json_path, 'w') as fp_temp:
                json.dump(_convert_numpy(data_to_write), fp_temp, indent=True)
            
            print(f"\n[ERROR] Failed to write to {json_path}. Data saved to temporary file: {temp_json_path}")
            print(f"Original Write Error: {e}")

    finally:
        # 3. RELEASE LOCK (Always released)
        if lock is not None: 
            lock.release()
            
    return data_to_write


def log_to_pickle(pickle_path, log, lock=None, dw_func=None):
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        file = open(pickle_path, 'wb')
        pickle.dump(log, file)
        file.close()
        if dw_func is not None: dw_func(pickle_path)
    finally:
        if lock is not None: lock.release()
    return


def update_log(json_path, new_output, start_id, policies, sort_log=True, lock=None):
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        try:
            new_logs = read_json(json_path, lock=None)
        except json.JSONDecodeError:
            new_logs = [] if 'full' in json_path else {}

        for id, log in enumerate(new_output):
            for pol in policies:
                new_logs[start_id+id][pol] = log[pol]

            if sort_log:
                new_logs[start_id+id] = _sort_log(new_logs[start_id+id])
        with open(json_path, 'w') as fp:
            json.dump(new_logs, fp, indent=True)
    finally:
        if lock is not None: lock.release()
    return new_logs


def compose_dirpath(fun):
    def inner(home_dir, ndays, nbins, output_dir, area, *args, **kwargs):
        if not isinstance(nbins, Iterable):
            dir_path = os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{nbins}")
            return fun(dir_path, *args, **kwargs)

        dir_paths = []
        for gs in nbins:
            dir_paths.append(os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{gs}"))
        return fun(dir_paths, *args, **kwargs)
    return inner


@compose_dirpath
def load_log_dict(dir_paths, nsamples, show_incomplete=False, lock=None):
    assert len(dir_paths) == len(nsamples), f"Len of dir_paths and nsamples lists must be equal, not {len(dir_paths)} != {len(nsamples)}"
    logs = {}    
    for path, ns in zip(dir_paths, nsamples):
        gsize = int(os.path.basename(path).split("_")[1])
        logs[f'{gsize}'] = os.path.join(path, f"log_mean_{ns}N.json")
        if show_incomplete and ns > 1:
            counter = Counter()
            log_full = read_json(os.path.join(path, f"log_full_{ns}N.json"), lock)
            for run in log_full:
                counter.update(run.keys())

            for key, val in dict(counter).items():
                incomplete = False
                if ns - val > 0:
                    if not incomplete:
                        incomplete = not incomplete
                        print(f'graph {gsize} incomplete runs:')
                    print('-', key, "-", ns - val)
    return logs


@compose_dirpath
def output_stats(dir_path, nsamples, policies, keys, sort_log=True, print_output=False, lock=None):
    mean_filename = os.path.join(dir_path, f"log_mean_{nsamples}N.json")
    std_filename = os.path.join(dir_path, f"log_std_{nsamples}N.json")
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        if os.path.isfile(mean_filename):
            mean_dit = read_json(mean_filename, lock=None)
            std_dit = read_json(std_filename, lock=None)
        else:
            mean_dit = {}
            std_dit = {}

        log = os.path.join(dir_path, f"log_full_{nsamples}N.json")
        data = read_json(log, lock=None)
        for pol in policies:
            tmp = []
            for n_id in range(nsamples):
                tmp.append(data[n_id][pol].values())
            mean_dit[pol] = {key: log for key, log in zip(keys, [*map(statistics.mean, zip(*tmp))])}
            std_dit[pol] = {key: log for key, log in zip(keys, [*map(statistics.stdev, zip(*tmp))])}

        if sort_log:
            mean_dit = _sort_log(mean_dit)
            std_dit = _sort_log(std_dit)
        if print_output:
            for lg, lg_std, pol in zip(mean_dit.values(), std_dit.values(), mean_dit.keys()):
                logm = lg.values() if isinstance(lg, dict) else lg
                logs = lg_std.values() if isinstance(lg_std, dict) else lg_std
                tmp_lg = [(str(x), str(y)) for x, y in zip(logm, logs)]
                if pol in policies:
                    print(f"{pol}:")
                    for (x, y), key in zip(tmp_lg, keys):
                        print(f"- {key} value: {x[:x.find('.')+3]} +- {y[:y.find('.')+5]}")

        with open(mean_filename, 'w') as fp:
            json.dump(mean_dit, fp, indent=True)
        with open(std_filename, 'w') as fp:
            json.dump(std_dit, fp, indent=True)
    finally:
        if lock is not None: lock.release()
    return mean_dit, std_dit


@compose_dirpath
def runs_per_policy(dir_paths, nsamples, policies, print_output=False, lock=None):
    assert len(dir_paths) == len(nsamples), f"Len of dir_paths and nsamples lists must be equal, not {len(dir_paths)} != {len(nsamples)}"
    runs_ls = []
    for path, ns in zip(dir_paths, nsamples):
        dit = {pol: [] for pol in policies}
        log = os.path.join(path, f"log_full_{ns}N.json")
        data = read_json(log, lock)
        for id, run_data in enumerate(data):
            for key in dit:
                if key in run_data:
                    dit[key].append(id)

        runs_ls.append(dit)
        if print_output:
            gsize = int(os.path.basename(path).rsplit('_', 1)[1])
            print(f"graph {gsize} #runs per policy:")
            for key, val in dit.items():
                print(f"- {key}: {len(val)}")
                print(" -- sample IDs:", val)
    return runs_ls


@compose_dirpath
def plot_attention_maps_wrapper(
    dir_path, attention_dict, model_name, execution_function,
    layer_idx=0, sample_idx=0, head_idx=0, batch_idx=0, x_labels=None, y_labels=None, **execution_kwargs
    ):
    """
    Plot attention maps as heatmaps for a given layer, head, batch, and simulation sample.
    
    Args:
        dir_path (str): Directory path to save the heatmap image.
        attention_dict (dict): Dictionary where:
                              - Keys are model names (str);
                              - Values are lists of attention data for each sample, where each element is a dictionary containing:
                                'attention_weights' tensor of shape [num_layers, n_heads, batch_size, graph_size, graph_size].
        model_name (str): Name of the model to extract attention maps for.
        execution_function (function): Function that handles the plotting/saving logic.
        layer_idx (int): Index of the layer to visualize.
        sample_idx (int): Index of the simulation sample to visualize.
        head_idx (int): Index of the head to visualize (-1 for average over all heads).
        batch_idx (int): Index of the data batch to visualize (-1 for average over all batches).
        x_labels (list, optional): Custom labels for x-axis vertices.
        y_labels (list, optional): Custom labels for y-axis vertices.
        **execution_kwargs: Additional arguments to pass to the execution function.
    
    Returns:
        attn_map (np.ndarray): The attention map as a Numpy array.
    """
    assert sample_idx >= 0, f"sample_idx {sample_idx} must be a non-negative integer"

    attention_weights = attention_dict[model_name][sample_idx]['attention_weights']
    assert layer_idx < attention_weights.shape[0], f"layer_idx {layer_idx} exceeds number of layers {attention_weights.shape[0]}"
    assert head_idx < attention_weights.shape[1], f"head_idx {head_idx} exceeds number of heads {attention_weights.shape[1]}"
    assert batch_idx < attention_weights.shape[2], f"layer_idx {batch_idx} exceeds batch size {attention_weights.shape[2]}"

    # Extract attention map
    if head_idx >= 0:
        if batch_idx >= 0:
            attn_map = attention_weights[layer_idx, head_idx, batch_idx].cpu().numpy()
            title = 'Attention Map (Layer {}, Head {}, Batch {})'.format(layer_idx, head_idx, batch_idx)
            attention_filename = os.path.join(dir_path, 'attention_maps', model_name, f'layer{layer_idx}_head{head_idx}_map{sample_idx}.png')
        else:
            attn_map = attention_weights[layer_idx, head_idx, :].mean(dim=0).cpu().numpy() # Average over batches
            title = 'Attention Map Average Over All Batches (Layer {}, Head {})'.format(layer_idx, head_idx)
            attention_filename = os.path.join(dir_path, 'attention_maps', model_name, f'layer{layer_idx}_head{head_idx}_map{sample_idx}.png')
    else:
        if batch_idx >= 0:
            attn_map = attention_weights[layer_idx, :, batch_idx].mean(dim=0).cpu().numpy() # Average over heads
            title = 'Attention Map Average Over All Heads (Layer {}, Batch {})'.format(layer_idx, batch_idx)
            attention_filename = os.path.join(dir_path, 'attention_maps', model_name, f'layer{layer_idx}_headavg_map{sample_idx}.png')
        else:
            attn_map = attention_weights[layer_idx, :, :].mean(dim=(0, 1)).cpu().numpy() # Average over heads and batches
            title = 'Attention Map Average Over All Heads and Batches (Layer {})'.format(layer_idx)
            attention_filename = os.path.join(dir_path, 'attention_maps', model_name, f'layer{layer_idx}_headavg_map{sample_idx}.png')

    try:
        os.makedirs(os.path.dirname(attention_filename), exist_ok=True)
    except Exception:
        raise Exception("directories to save attention maps do not exist and could not be created")

    # Dynamically set figure size based on map_size
    base_vertexsize = 0.5
    map_size = math.isqrt(attn_map.shape[0] * attn_map.shape[1])
    min_figsize = 6.0
    max_figsize = 30.0
    figsize = min(max(min_figsize, base_vertexsize * map_size), max_figsize)
    fig = plt.figure(figsize=(figsize, figsize))

    # Adjust annotations and font sizes to scale inversely with map_size
    max_ticsize = 8
    max_annotsize = 8
    annot = True if map_size <= 55 else False  # Disable annotations for large graphs to avoid clutter
    tick_fontsize = max(max_ticsize, 14 - map_size // 10)
    annot_fontsize = max(max_annotsize, 12 - map_size // 10)
    
    # Plot and/or log attention heatmap
    plt.title(title)
    sns.heatmap(attn_map, annot=annot, cmap='viridis', fmt='.2f', cbar=True, annot_kws={'fontsize': annot_fontsize})
    plt.xlabel('Key Vertices')
    plt.ylabel('Query Vertices')
    if x_labels is None: x_labels = [f'Vertex {i}' for i in range(attn_map.shape[0])]
    if y_labels is None: y_labels = [f'Vertex {i}' for i in range(attn_map.shape[1])]
    plt.xticks(ticks=range(attn_map.shape[0]), labels=x_labels, rotation=45, fontsize=tick_fontsize)
    plt.yticks(ticks=range(attn_map.shape[1]), labels=y_labels, rotation=0, fontsize=tick_fontsize)
    plt.tight_layout()
    execution_function(
        plot_target=attn_map,
        fig=fig,
        title=title,
        figsize=figsize,
        x_labels=x_labels,
        y_labels=y_labels,
        fig_filename=attention_filename,
        **execution_kwargs
    )
    return attn_map


def visualize_interactive_plot(**kwargs):
    """Execution function for interactive visualization"""
    interactive_fig = px.imshow(kwargs['plot_target'], text_auto='.2f', color_continuous_scale='Viridis', title=kwargs['title'], 
                                labels={'x': kwargs['x_labels'], 'y': kwargs['y_labels']}, width=kwargs['figsize'], height=kwargs['figsize'])
    interactive_fig.update_xaxes(tickvals=list(range(len(kwargs['x_labels']))), ticktext=kwargs['x_labels'])
    interactive_fig.update_yaxes(tickvals=list(range(len(kwargs['y_labels']))), ticktext=kwargs['y_labels'])
    interactive_fig.show()
    return


def log_plot(visualize=False, **kwargs):
    """Execution function for saving static plots"""
    kwargs['fig'].savefig(kwargs['fig_filename'], bbox_inches='tight')
    if visualize: plt.show()
    plt.close(kwargs['fig'])
    return