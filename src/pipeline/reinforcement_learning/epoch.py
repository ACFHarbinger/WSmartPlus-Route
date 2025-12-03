import os
import json
import time
import math
import torch
import pandas as pd

from tqdm import tqdm
from ..simulator.bins import Bins
from src.utils.functions import move_to
from src.utils.data_utils import generate_waste_prize


def set_decode_type(model, decode_type):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def get_inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    def _eval_model_bat(bat):
        with torch.no_grad():
            ucost, _, _, _ = model(move_to(bat, opts['device']), cost_weights=None)
        return ucost.data.cpu()

    set_decode_type(model, "greedy")
    model.eval()    
    if opts['temporal_horizon'] > 0 and opts['model'] in ['tam']:
        dataset.fill_history = torch.zeros((opts['val_size'], opts['graph_size'], opts['temporal_horizon']))
        dataset.fill_history[:, :, -1] = torch.stack([instance['waste'] for instance in dataset.data])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts['eval_batch_size'], pin_memory=True)
    costs = []
    for bat_id, bat in enumerate(tqdm(dataloader, disable=opts['no_progress_bar'])):
        bat = prepare_batch(bat, bat_id, dataset, dataloader, opts)      
        cost = _eval_model_bat(bat)
        costs.append(cost)
    return torch.cat(costs, 0)


# TODO: change way new costs are calculated
def validate_update(model, dataset, cw_dict, opts):
    def _eval_model_bat(bat, dist_matrix):
        with torch.no_grad():
            ucost, c_dict, attn_dict = get_inner_model(model).compute_batch_sim(move_to(bat, opts['device']), move_to(dist_matrix, opts['device']))
        return ucost, c_dict, attn_dict

    set_decode_type(model, "greedy")
    model.eval()
    if opts['temporal_horizon'] > 0 and opts['model'] in ['tam']:
        dataset.fill_history = torch.zeros((opts['val_size'], opts['graph_size'], opts['temporal_horizon']))
        dataset.fill_history[:, :, -1] = torch.stack([instance['waste'] for instance in dataset.data])

    all_costs = {'overflows': [], 'kg': [], 'km': []}
    all_ucosts = move_to(torch.tensor([]), opts['device'])
    attention_dict = {'attention_weights': [], 'graph_masks': []}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts['eval_batch_size'], pin_memory=True)
    print('Validating...')
    for bat_id, bat in enumerate(tqdm(disable=opts['no_progress_bar'])):
        bat = prepare_batch(bat, bat_id, dataset, dataloader, opts)
        ucost, cost_dict, attn_dict = _eval_model_bat(bat, dataset.dist_matrix)
        for key in attention_dict.keys():
            attention_dict[key].append[attn_dict[key]]

        all_ucosts = torch.cat((all_ucosts, ucost), 0)
        for key, val in cost_dict.items():
            all_costs[key].append(val)

    for key, val in attention_dict.items():
        attention_dict[key] = torch.cat(val)

    for key, val in all_costs.items():
        all_costs[key] = torch.cat(val)

    eps = 1e-8
    overflows_mean = torch.mean(all_costs['overflows'].float())
    kg_mean = torch.mean(all_costs['kg'].float())
    km_mean = torch.mean(all_costs['km'].float())
    efficiency = kg_mean / km_mean.clamp(min=eps)

    # Calculate waste (kg_lost) if available, otherwise estimate
    kg_lost = torch.tensor(0.0)
    if 'kg_lost' in all_costs:
        kg_lost = torch.mean(all_costs['kg_lost'])
    else:
        kg_lost = overflows_mean * 100 - kg_mean
    
    # Calculate target ratios based on the better performing model
    target_efficiency = 12.5  # Target kg/km based on the better model
    target_overflow_ratio = 1.5  # Target overflow level compared to the better model
    
    # Calculate how far current metrics are from targets
    efficiency_gap = (target_efficiency - efficiency) / target_efficiency
    overflow_gap = (overflows_mean - 372.5) / 372.5 
    waste_ratio = kg_lost / kg_mean
    
    # Set max adaptation rate to prevent extreme weight changes
    max_adaptation = min(opts['adaptation_rate'], 0.2)
    
    # Calculate adjustment factors with bounded sigmoid to prevent runaway updates
    def bounded_sigmoid(x):
        return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0
    
    # Proportional adjustments capped by max_adaptation
    overflow_adjust = max_adaptation * bounded_sigmoid(overflow_gap)
    efficiency_adjust = max_adaptation * bounded_sigmoid(efficiency_gap)
    waste_adjust = -max_adaptation * bounded_sigmoid(waste_ratio * 5.0)  # Penalize waste more aggressively
    
    # Baseline weight distribution (thirds if no previous history)
    constraint_third = opts['constraint'] / 3
    
    # Update weights with damping to prevent oscillation
    damping = 0.7  # Damping factor to prevent oscillation (0.7 means 70% new, 30% old)
    
    overflow_w = cw_dict['overflows'] * (1 - damping) + (cw_dict['overflows'] * (1 + overflow_adjust)) * damping
    waste_w = cw_dict['waste'] * (1 - damping) + (cw_dict['waste'] * (1 + waste_adjust)) * damping
    length_w = cw_dict['length'] * (1 - damping) + (cw_dict['length'] * (1 + efficiency_adjust)) * damping
    
    # Enforce minimum and maximum bounds on weights
    min_weight = 0.05 * opts['constraint']
    max_weight = 0.6 * opts['constraint']  # Prevent any single weight from dominating
    
    overflow_w = overflow_w.clamp(min_weight, max_weight).item()
    waste_w = waste_w.clamp(min_weight, max_weight).item()
    length_w = length_w.clamp(min_weight, max_weight).item()
    
    # Normalize weights to sum to constraint
    sum_w = overflow_w + waste_w + length_w
    constraint_factor = opts['constraint'] / sum_w
    
    new_cw = {
        'overflows': overflow_w * constraint_factor,
        'waste': waste_w * constraint_factor,
        'length': length_w * constraint_factor
    }
    
    # Safety check: ensure waste weight doesn't grow too much
    if new_cw['waste'] > 1.5 * cw_dict['waste']:
        # Cap waste weight growth
        new_cw['waste'] = 1.5 * cw_dict['waste']
        # Redistribute remaining weight
        remaining = opts['constraint'] - new_cw['waste']
        overflow_length_ratio = overflow_w / (overflow_w + length_w)
        new_cw['overflows'] = remaining * overflow_length_ratio
        new_cw['length'] = remaining * (1 - overflow_length_ratio)
     
    """max_val = max(kg_mean, km_mean, overflows_mean)
    norm_efficiency = efficiency / max_val.clamp(min=eps)
    norm_overflow = overflows_mean / max_val.clamp(min=eps)
    overflow_efficiency = norm_overflow / norm_efficiency.clamp(min=eps)
    
    kg_adjust = opts['adaptation_rate'] * (torch.tanh(efficiency - 1.0))
    km_adjust = opts['adaptation_rate'] * (-torch.tanh(efficiency - 1.0))
    overflow_adjust = opts['adaptation_rate'] * torch.tanh(overflow_efficiency - 1.0)
    
    overflow_prev_factor = -0.5 * opts['adaptation_rate'] * ((cw_dict['overflows'] - opts['constraint']/3) / (opts['constraint']/3))
    waste_prev_factor = -0.5 * opts['adaptation_rate'] * ((cw_dict['waste'] - opts['constraint']/3) / (opts['constraint']/3))
    length_prev_factor = -0.5 * opts['adaptation_rate'] * ((cw_dict['length'] - opts['constraint']/3) / (opts['constraint']/3))

    overflow_w = overflow_adjust + overflow_prev_factor + cw_dict['overflows']
    waste_w = kg_adjust + waste_prev_factor + cw_dict['waste']
    length_w = km_adjust + length_prev_factor + cw_dict['length']

    min_weight = 0.05 * opts['constraint']
    overflow_w = min_weight + (overflow_w - min_weight).clamp(min=0).item()
    waste_w = min_weight + (waste_w - min_weight).clamp(min=0).item()
    length_w = min_weight + (length_w - min_weight).clamp(min=0).item()

    sum_w = overflow_w + waste_w + length_w
    new_cw = {
        'overflows': overflow_w * (opts['constraint'] / sum_w),
        'waste': waste_w * (opts['constraint'] / sum_w),
        'length': length_w * (opts['constraint'] / sum_w)
    }"""

    print("New cost function weights: ")
    for key, val in new_cw.items():
        print(f"- {key}: {val}")

    avg_cost = all_ucosts.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(all_ucosts) / math.sqrt(len(all_ucosts))))
    
    for key, val in all_costs.items():
        val = val.float()
        print('- {}: {} +- {}'.format(
        key, val.mean(), torch.std(val) / math.sqrt(len(val))))
    return new_cw, avg_cost, all_costs
    """# Print additional metrics for monitoring
    print('\nDerived metrics:')
    print(f'Efficiency (kg/km): {efficiency.item():.4f} (target: {target_efficiency:.4f})')
    print(f'Overflow count: {overflows_mean.item():.2f}')
    print(f'Weight adjustments - Overflow: {overflow_adjust.item():.4f}, Waste: {waste_adjust.item():.4f}, Length: {efficiency_adjust.item():.4f}')
    
    return new_cw, avg_cost, all_costs"""


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def prepare_epoch(optimizer, epoch, problem, tb_logger, cost_weights, opts):
    print("Start train {} {}, lr={} for run {}".format(
        "day" if opts['train_time'] else "epoch", epoch, optimizer.param_groups[0]['lr'], opts['run_name']))
    step = epoch * (opts['epoch_size'] // opts['batch_size'])
    if not opts['no_tensorboard']:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
    
    loss_keys = list(cost_weights.keys()) + ['total', 'nll', 'reinforce_loss'] if opts['baseline'] is None \
        else list(cost_weights.keys()) + ['total', 'nll', 'reinforce_loss', 'baseline_loss']
    if opts['train_time'] and opts['train_dataset'] is not None:
        training_dataset = problem.make_dataset(
            filename=opts['train_dataset'], area=opts['area'], waste_type=opts['waste_type'],
            distribution=opts['data_distribution'], vertex_strat=opts['vertex_method'],
            dist_matrix_path=opts['dm_filepath'], dist_strat=opts['distance_method'],
            number_edges=opts['edge_threshold'], edge_strat=opts['edge_method'],
            focus_graph=opts['focus_graph'], focus_size=opts['focus_size'],
            size=opts['graph_size'], num_samples=opts['epoch_size']
        )
    elif not opts['train_time'] and opts['train_dataset'] is not None:
        filename = os.path.join("data", "datasets", problem.NAME, 
            "{}{}{}_{}{}_seed{}.pkl".format(problem.NAME, opts['graph_size'],
            "_{}".format(opts['data_distribution']) if opts['data_distribution'] is not None else "",
            opts['train_dataset'], epoch, opts['seed']))
        training_dataset = problem.make_dataset(
            filename=filename, number_edges=opts['edge_threshold'], edge_strat=opts['edge_method'], size=opts['graph_size'],
            dist_matrix_path=opts['dm_filepath'], num_samples=opts['epoch_size'], dist_strat=opts['distance_method'],
            area=opts['area'], distribution=opts['data_distribution'], vertex_strat=opts['vertex_method'], 
            waste_type=opts['waste_type'], focus_graph=opts['focus_graph'], focus_size=opts['focus_size']
        )
    else:
        training_dataset = problem.make_dataset(
            size=opts['graph_size'], num_samples=opts['epoch_size'], dist_strat=opts['distance_method'],
            distribution=opts['data_distribution'], vertex_strat=opts['vertex_method'], area=opts['area'],
            number_edges=opts['edge_threshold'], edge_strat=opts['edge_method'], focus_size=opts['focus_size'],
            waste_type=opts['waste_type'], focus_graph=opts['focus_graph'], dist_matrix_path=opts['dm_filepath']
        )
    return step, training_dataset, loss_keys


def prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts):
    if opts['problem'] in ['vrpp', 'wcrp', 'cwcvrp', 'sdwcvrp'] and opts['data_distribution'] == 'emp':
        data_dir = os.path.join(os.getcwd(), "data", "wsr_simulator")
        with open(os.path.join(data_dir, 'bins_selection', opts['focus_graph'])) as js:
            idx = json.load(js)
        bins = Bins(opts['graph_size'], data_dir, sample_dist=opts['data_distribution'], area=opts['area'], indices=idx[0], grid=None)
    else:
        idx = [None]
        bins = None

    step, training_dataset, loss_keys = prepare_epoch(optimizer, day, problem, tb_logger, cost_weights, opts)
    if opts['temporal_horizon'] > 0:
        data_size = training_dataset.size
        graphs = torch.stack([torch.cat((x['depot'].unsqueeze(0), x['loc'])) for x in training_dataset])
        if opts['problem'] in ['vrpp', 'wcrp', 'cwcvrp', 'sdwcvrp']:
            for day_id in range(1, opts['temporal_horizon'] + 1):
                training_dataset["fill{}".format(day_id)] = torch.from_numpy(generate_waste_prize(opts['graph_size'], opts['data_distribution'], graphs, data_size, bins)).float()

            if opts['model'] in ['tam']:
                training_dataset.fill_history = torch.zeros((opts['epoch_size'], opts['graph_size'], opts['temporal_horizon'])).float()
                training_dataset.fill_history[:, :, -1] = torch.stack([instance['waste'] for instance in training_dataset.data])

    # Setup for logging
    stat_keys = ['mean', 'std', 'min', 'max']
    col_multi_index = pd.MultiIndex.from_product([loss_keys, stat_keys])
    #day_col = pd.MultiIndex.from_tuples([('day', '')])
    #all_columns = day_col.append(col_multi_index)
    table_df = pd.DataFrame(columns=col_multi_index)
    return step, training_dataset, loss_keys, table_df, (bins,)


def complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, epoch, step, epoch_duration, tb_logger, cost_weights, opts):
    print("Finished {} {}, took {} s".format(
        "day" if opts['train_time'] else "epoch", epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    if (opts['checkpoint_epochs'] != 0 and epoch % opts['checkpoint_epochs'] == 0) or epoch == opts['n_epochs'] - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts['save_dir'], 'epoch-{}.pt'.format(epoch))
        )

    if opts['val_size'] > 0:
        avg_reward = validate(model, val_dataset, opts)
        #new_cw, avg_cost, _ = validate_update(model, val_dataset, cost_weights, opts)
        if not opts['no_tensorboard']:
            tb_logger.log_value('val_avg_cost', avg_reward, step)

    baseline.epoch_callback(model, epoch)
    lr_scheduler.step() # lr_scheduler should be called at end of epoch
    if opts['device'] == "cuda":
        torch.cuda.empty_cache()
    return None


def prepare_batch(batch, batch_id, dataset, dataloader, opts, day=1):
    if opts['model'] in ['tam'] and opts['temporal_horizon'] > 0:
        batch_size = dataloader.batch_size
        start_idx = batch_id * batch_size
        end_idx = min((batch_id + 1) * batch_size, len(dataset))
        batch_idx = torch.arange(start_idx, end_idx)
        batch['fill_history'] = dataset.fill_history[batch_idx]
    
    counter = 0
    filldays = ['fill{}'.format(day_id) for day_id in range(day, day + opts['temporal_horizon'])]
    for k, v in batch.items():
        if 'fill' in k:
            if k in filldays:
                counter += 1
                batch['fill{}'.format(counter)] = v
        else:
            batch[k] = v
    if opts['focus_graph'] is not None:
        if opts['encoder'] in ['gac', 'tgc']:
            batch['edges'] = dataset.edges.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1).float()
        else:
            batch['edges'] = dataset.edges.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1).bool()
    batch['dist'] = dataset.dist_matrix.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1)
    return batch


def update_time_dataset(model, optimizer, dataset, routes, day, opts, args):
    data_size = dataset.size
    routes = torch.stack(routes).contiguous().view(-1, routes[0].size(1))
    graphs = torch.stack([torch.cat((x['depot'].unsqueeze(0), x['loc'])) for x in dataset])
    
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    if opts['problem'] in ['vrpp', 'wcrp']:
        # Get masks for bins present in routes
        sorted_routes = routes.sort(1)[0]
        dataset_dim, node_dim = sorted_routes.size()
        visited_mask = torch.zeros((dataset_dim, node_dim), dtype=torch.bool).to(sorted_routes.device)
        col_idx = sorted_routes[sorted_routes != 0]
        row_idx = torch.arange(dataset_dim, device=sorted_routes.device).repeat_interleave((sorted_routes != 0).sum(dim=1))
        visited_mask[row_idx, col_idx] = True
        
        # Set waste in visited bins to 0 and remove waste above max_waste
        waste = torch.stack([torch.cat((torch.tensor([0]), x['waste'])) for x in dataset])
        max_waste = torch.stack([x['max_waste'] for x in dataset]).unsqueeze(1)
        waste[visited_mask] = 0
        waste = waste[:, 1:].clamp_(max=max_waste)

        # Add daily waste filling
        fillday = f'fill{day}'
        if opts['temporal_horizon'] > 0:
            waste += torch.stack([x[fillday] for x in dataset])
            for day_id in range(day + 1, day + opts['temporal_horizon'] + 1):
                fill_key = "fill{}".format(day_id)
                if day_id > opts['n_epochs']:
                    dataset[fill_key] = torch.zeros((dataset_dim, node_dim), dtype=torch.float).to(sorted_routes.device)
                elif fill_key in dataset[0].keys():
                    dataset[fill_key] = torch.stack([x[fill_key] for x in dataset])
                else:
                    dataset[fill_key] = torch.from_numpy(generate_waste_prize(opts['graph_size'], opts['data_distribution'], graphs, data_size, *args)).float()
            if opts['model'] in ['tam']:
                dataset.fill_history = get_inner_model(model).update_fill_history(dataset.fill_history, fill)
        else:
            if fillday in dataset[0].keys():
                fill = torch.stack([x[fillday] for x in dataset])
            else:
                fill = torch.from_numpy(generate_waste_prize(opts['graph_size'], opts['data_distribution'], graphs, data_size, *args)).float()
            waste += fill
        dataset['waste'] = torch.clone(waste).to(dtype=torch.float)
    is_val = data_size == opts['val_size'] 
    print("Start {} day {},{} for run {}".format("eval" if is_val else "train", day, 
        " lr={}".format(optimizer.param_groups[0]['lr']) if not is_val else "", opts['run_name']))
    return dataset
