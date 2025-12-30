import os
import json
import torch
import numpy as np
import multiprocessing as mp
import torch.nn.functional as F

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


# Attention, Learn to Solve Routing Problems
def get_inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_problem(name):
    from logic.src.problems import VRPP, CVRPP, WCVRP, CWCVRP, SDWCVRP
    problem = {
        'vrpp': VRPP,
        'cvrpp': CVRPP,
        'wcvrp': WCVRP,
        'cwcvrp': CWCVRP,
        'sdwcvrp': SDWCVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def load_data(load_path, resume):
    load_data = {}
    assert load_path is None or resume is None, \
    "Only one of load path and resume can be given"
    
    load_path = load_path if load_path is not None \
                else resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    return load_data


def move_to(var, device):
    if var is None:
        return None
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""
    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)
    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl in ("vrpp", "wcvrp"):
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    
    if 'aggregation_graph' not in args:
        args['aggregation_graph'] = "avg"
    return args


def load_model(path, epoch=None):
    from logic.src.models import (
        GraphAttentionEncoder, GraphAttConvEncoder, TransGraphConvEncoder, GatedGraphAttConvEncoder,
        AttentionModel, TemporalAttentionModel, DeepDecoderAttentionModel
    )
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            pt_files = [f for f in os.listdir(path) if f.endswith('.pt')]
            epochs = []
            for f in pt_files:
                name = os.path.splitext(f)[0]
                if '-' in name:
                    parts = name.split('-')
                    if len(parts) == 2 and parts[0] == 'epoch' and parts[1].isdigit():
                        epochs.append(int(parts[1]))
            
            if not epochs:
                raise ValueError("No valid epoch files (epoch-N.pt) found in directory: {}".format(path))
            epoch = max(epochs)
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))
    problem = load_problem(args['problem'])
    encoder_class = {
        'gat': GraphAttentionEncoder,
        'gac': GraphAttConvEncoder,
        'tgc': TransGraphConvEncoder,
        'ggac': GatedGraphAttConvEncoder
    }.get(args.get('encoder', 'gat'), None)
    assert encoder_class is not None, "Unknown encoder: {}".format(encoder_class)

    model_class = {
        'am': AttentionModel,
        'tam': TemporalAttentionModel,
        'ddam': DeepDecoderAttentionModel
    }.get(args.get('model', 'am'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        encoder_class,
        args['n_encode_layers'],
        args['n_encode_sublayers'],
        args['n_decode_layers'],
        n_heads=args['n_heads'],
        normalization=args['normalization'],
        norm_learn_affine=args['learn_affine'],
        norm_track_stats=args['track_stats'],
        norm_eps_alpha=args['epsilon_alpha'],
        norm_momentum_beta=args['momentum_beta'],
        lrnorm_k=args['lrnorm_k'],
        gnorm_groups=args['gnorm_groups'],
        activation_function=args['activation'],
        af_param=args['af_param'],
        af_threshold=args['af_threshold'],
        af_replacement_value=args['af_replacement'],
        af_num_params=args['af_nparams'],
        af_uniform_range=args['af_urange'],
        dropout_rate=args['dropout'],
        aggregation=args['aggregation'],
        aggregation_graph=args['aggregation_graph'],
        tanh_clipping=args['tanh_clipping'],
        mask_inner=args.get('mask_inner', True),
        mask_logits=args.get('mask_logits', True),
        mask_graph=args.get('mask_graph', False),
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None),
        temporal_horizon=args.get('temporal_horizon', 0),
        spatial_bias=args.get('spatial_bias', False),
        spatial_bias_scale=args.get('spatial_bias_scale', 1.0),
        entropy_weight=args.get('entropy_weight', 0.0),
        predictor_layers=args.get('n_predict_layers', None)
    )

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model, *_ = _load_model_file(model_filename, model)
    model.eval()  # Put in eval mode
    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test_sim', *dataset[0]))
    # return [res]
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus
    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0

    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (mp.Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def get_path_until_string(path, end_str):
    path_ls = str.split(path, os.sep)
    try:
        idx = path_ls.index(end_str)
        return os.sep.join(path_ls[:idx+1])
    except ValueError as ve:
        print(f"Path '{path}' does not contain '{end_str}'")
        return None


# Tensor functions
def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches
    :param n: the total number of elements, optional if it cannot be determined as args[0].size(0)
    :param f: The function that is computed, should take only tensors as arguments and return tensor or tuple of tensors
    :param calc_batch_size: The batch size to use when computing this function
    :param args: Tensor arguments with equally sized first batch dimension
    :return: f(*args), this should be one or multiple tensors with equally sized first batch dimension
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    # all_res = [f(*batch_args) for batch_args in zip(*[torch.chunk(arg, n_batches) for arg in args])]
    # We do not use torch.chunk such that it also works for other classes that support slicing
    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)


def add_attention_hooks(model_module):
        graph_masks = []
        attention_weights = []
        
        def hook(module, input, output):
            if hasattr(module, 'last_attn') and module.last_attn is not None:
                graph_masks.append(module.last_attn[-1])
                attention_weights.append(module.last_attn[0])
        
        # Register hooks on all MHA layers
        hook_data = {
            'weights': attention_weights,
            'masks': graph_masks,
            'handles': []
        }
        for layer in model_module.layers:
            # Get the actual attention module (skip the SkipConnection wrapper), if layer has attention
            if not hasattr(layer, 'att'): continue
            attention_module = layer.att.module
            
            # Register hook and store the handle
            hook_handle = attention_module.register_forward_hook(hook)
            hook_data['handles'].append(hook_handle)
        return hook_data


# Sampling functions
def do_batch_rep(v, n):
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)
    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)
    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)

        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)

    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    return minpis, mincosts
