import os
import json
import torch
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


# Attention, Learn to Solve Routing Problems
def get_inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_problem(name):
    from problems import TSP, PDP, CVRP, SDVRP, OP, VRPP, WCRP, CWCVRP, SDWCVRP, PCTSPDet, PCTSPStoch
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'vrpp': VRPP,
        'wcrp': WCRP,
        'cwcvrp': CWCVRP,
        'sdwcvrp': SDWCVRP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
        'pdp': PDP,
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
        if probl in ("op", "wcrp"):
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    
    if 'aggregation_graph' not in args:
        args['aggregation_graph'] = "avg"
    return args


def load_model(path, epoch=None):
    from models import (
        GraphAttentionEncoder, GraphAttConvEncoder, TransGraphConvEncoder,
        AttentionModel, PointerNetwork, TemporalAttentionModel, DeepDecoderAttentionModel
    )
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))
    problem = load_problem(args['problem'])
    encoder_class = {
        'gat': GraphAttentionEncoder,
        'gac': GraphAttConvEncoder,
        'tgc': TransGraphConvEncoder
    }.get(args.get('encoder', 'gat'), None)
    assert encoder_class is not None, "Unknown encoder: {}".format(encoder_class)

    model_class = {
        'am': AttentionModel,
        'pn': PointerNetwork,
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