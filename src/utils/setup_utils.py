import os
import torch
import gurobipy as gp
import torch.optim as optim

from .definitions import ROOT_DIR
from dotenv import dotenv_values
from src.utils.crypto_utils import decrypt_file_data
from src.utils.functions import load_model, get_inner_model


def setup_cost_weights(opts, def_val=1.):
    def _set_val(cost_weight, default_value):
        return default_value if cost_weight is None else cost_weight
    #def _set_weight(opts, cost_weight, default_value=1.):
    #return opts.get(cost_weight, default_value)

    cw_dict = {}
    if opts['problem'] == 'wcrp':
        #cw_dict['lost'] = opts['w_lost'] = _set_val(opts['w_lost'], def_val)
        cw_dict['waste'] = opts['w_waste'] = _set_val(opts['w_waste'], def_val)
        cw_dict['length'] = opts['w_length'] = _set_val(opts['w_length'], def_val)
        cw_dict['overflows'] = opts['w_overflows'] = _set_val(opts['w_overflows'], def_val)
    elif opts['problem'] == 'vrpp':
        cw_dict['waste'] = opts['w_waste'] = _set_val(opts['w_waste'], def_val)
        cw_dict['length'] = opts['w_length'] = _set_val(opts['w_length'], def_val)
    elif opts['problem'] in ['cwcvrp', 'sdwcvrp']:
        cw_dict['lost'] = opts['w_lost'] = _set_val(opts['w_lost'], def_val)
        cw_dict['length'] = opts['w_length'] = _set_val(opts['w_length'], def_val)
        cw_dict['overflows'] = opts['w_overflows'] = _set_val(opts['w_overflows'], def_val)
    elif opts['problem'] == 'op':
        cw_dict['prize'] = opts['w_prize'] = _set_val(opts['w_prize'], def_val)
    elif opts['problem'] in ['tsp', 'cvrp', 'sdvrp', 'pdp']:
        cw_dict['length'] = opts['w_length'] = _set_val(opts['w_length'], def_val)
    elif opts['problem'] == 'pctsp':
        cw_dict['prize'] = opts['w_prize'] = _set_val(opts['w_prize'], def_val)
        cw_dict['length'] = opts['w_length'] = _set_val(opts['w_length'], def_val)
        cw_dict['penalty'] = opts['w_penalty'] = _set_val(opts['w_penalty'], def_val)
    return cw_dict


def setup_model(policy, general_path, device, lock, temperature=1, decode_type="greedy"):
    def _load_model(general_path, model_name, device, temperature, decode_type, lock):
        model_path = os.path.join(general_path, model_name)
        with lock:
            model, configs = load_model(model_path)
        
        model.to(device)
        model.eval()
        model.set_decode_type(decode_type, temp=temperature)
        return model, configs

    if 'amgc' in policy:
        return _load_model(general_path, "amgc", device, temperature, decode_type, lock)
    elif 'am' in policy:
        return _load_model(general_path, "am", device, temperature, decode_type, lock)
    elif 'transgcn' in policy:
        return _load_model(general_path, "transgcn", device, temperature, decode_type, lock)
    return None


def setup_env(policy, server=False, gplic_filename=None, symkey_name=None, env_filename=None):
    if 'vrpp' in policy and 'hexaly' not in policy:
        if server:
            convert_int = lambda param: int(param) if param.isdigit() else param
            if gplic_filename is not None:
                gplic_path = os.path.join(ROOT_DIR, "assets", "api", gplic_filename)
                if symkey_name:
                    data = decrypt_file_data(gplic_path, symkey_name=symkey_name, env_filename=env_filename)
                else:
                    with open(gplic_path, 'r') as gp_file:
                        data = gp_file.read()
                params = {line.split('=')[0]: convert_int(line.split('=')[1]) for line in data.split('\n') if '=' in line}
            else:
                assert env_filename is not None
                env_path = os.path.join(ROOT_DIR, "env", env_filename)
                config = dotenv_values(env_path)
                glp_ls = ['WLSACCESSID', 'WLSSECRET', 'LICENSEID']
                params = {glp: convert_int(config.get(glp, '')) for glp in glp_ls}
                for glp_key, glp_val in params.items():
                    if isinstance(glp_val, str) and glp_val == '':
                        raise ValueError(f"Missing parameter {glp_key} for Gurobi license")
        else:
            params = {}
        params['OutputFlag'] = 0
        return gp.Env(params=params)
    

def setup_model_and_baseline(problem, data_load, use_cuda, opts):
    from src.models import (
        CriticNetwork, CriticNetworkLSTM,
        AttentionModel, PointerNetwork, TemporalAttentionModel,
        DeepDecoderAttentionModel, HierarchicalTemporalAttentionModel,
        GraphAttentionEncoder, GraphAttConvEncoder, TransGraphConvEncoder,
        NoBaseline, WarmupBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline
    )
    encoder_class = {
        'gat': GraphAttentionEncoder,
        'gac': GraphAttConvEncoder,
        'tgc': TransGraphConvEncoder
    }.get(opts['encoder'], None)
    assert encoder_class is not None, \
    "Unknown encoder: {}".format(encoder_class)

    model_class = {
        'am': AttentionModel,
        'pn': PointerNetwork,
        'tam': TemporalAttentionModel,
        'htam': HierarchicalTemporalAttentionModel,
        'ddam': DeepDecoderAttentionModel
    }.get(opts['model'], None)
    assert model_class is not None, \
    "Unknown model: {}".format(model_class)

    model = model_class(
        opts['embedding_dim'],
        opts['hidden_dim'],
        problem,
        encoder_class,
        opts['n_encode_layers'],
        opts['n_encode_sublayers'],
        opts['n_decode_layers'],
        n_heads=opts['n_heads'],
        normalization=opts['normalization'],
        norm_learn_affine=opts['learn_affine'],
        norm_track_stats=opts['track_stats'],
        norm_eps_alpha=opts['epsilon_alpha'],
        norm_momentum_beta=opts['momentum_beta'],
        lrnorm_k=opts['lrnorm_k'],
        gnorm_groups=opts['gnorm_groups'],
        activation_function=opts['activation'],
        af_param=opts['af_param'],
        af_threshold=opts['af_threshold'],
        af_replacement_value=opts['af_replacement'],
        af_num_params=opts['af_nparams'],
        af_uniform_range=opts['af_urange'],
        dropout_rate=opts['dropout'],
        aggregation=opts['aggregation'],
        aggregation_graph=opts['aggregation_graph'],
        tanh_clipping=opts['tanh_clipping'],
        mask_inner=opts['mask_inner'],
        mask_logits=opts['mask_logits'],
        mask_graph=opts['mask_graph'],
        checkpoint_encoder=opts['checkpoint_encoder'],
        shrink_size=opts['shrink_size'],
        temporal_horizon=opts['temporal_horizon'],
        predictor_layers=opts['n_predict_layers']
    ).to(opts['device'])

    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **data_load.get('model', {})})
    
    # Initialize baseline
    if opts['baseline'] == 'exponential':
        baseline = ExponentialBaseline(opts['exp_beta'])
    elif opts['baseline'] == 'critic' or opts['baseline'] == 'critic_lstm':
        assert opts['baseline'] != 'critic_lstm' or problem.NAME == 'tsp', \
        "Critic LSTM only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts['embedding_dim'],
                    opts['hidden_dim'],
                    opts['n_encode_layers'],
                    opts['tanh_clipping']
                )
                if opts['baseline'] == 'critic_lstm'
                else
                CriticNetwork(
                    problem,
                    encoder_class,
                    opts['embedding_dim'],
                    opts['hidden_dim'],
                    opts['n_encode_layers'],
                    opts['n_other_layers'],
                    opts['normalization'],
                    opts['activation'],
                )
            ).to(opts['device'])
        )
    elif opts['baseline'] == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts['baseline'] is None, \
        "Unknown baseline: {}".format(opts['baseline'])
        baseline = NoBaseline()

    if opts['bl_warmup_epochs'] > 0:
        baseline = WarmupBaseline(baseline, opts['bl_warmup_epochs'], 
                                warmup_exp_beta=opts['exp_beta'])

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in data_load:
        baseline.load_state_dict(data_load['baseline'])
    
    return model, baseline


def setup_optimizer_and_lr_scheduler(model, baseline, data_load, opts):
    optimizer_params = [{'params': model.parameters(), 'lr': opts['lr_model']}] + (
        [{'params': baseline.get_learnable_parameters(), 'lr': opts['lr_critic_value']}]
        if len(baseline.get_learnable_parameters()) > 0
        else []
    )
    optimizer = {
        'adam': optim.Adam(optimizer_params),
        'adamax': optim.Adamax(optimizer_params),
        'adamw': optim.AdamW(optimizer_params),
        'radam': optim.RAdam(optimizer_params),
        'nadam': optim.NAdam(optimizer_params),
        'sadam': optim.SparseAdam(optimizer_params),
        'adadelta': optim.Adadelta(optimizer_params),
        'adagrad': optim.Adagrad(optimizer_params),
        'rmsprop': optim.RMSprop(optimizer_params),
        'rprop': optim.Rprop(optimizer_params),
        'lbfgs': optim.LBFGS(optimizer_params),
        'asgd': optim.ASGD(optimizer_params),
        'sgd': optim.SGD(optimizer_params)
    }.get(opts['optimizer'], None)
    assert optimizer is not None, "Unknown optimizer: {}".format(opts['optimizer'])

    # Load optimizer state, make sure script is called with same type of optimizer
    if 'optimizer' in data_load:
        optimizer.load_state_dict(data_load['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts['device'])

    # Initialize learning rate scheduler!
    lr_scheduler = {
        'exp': optim.lr_scheduler.ExponentialLR(optimizer, opts['lr_decay']),
        'step': optim.lr_scheduler.StepLR(optimizer, opts['lrs_step_size'], opts['lr_decay']),
        'mult': optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: opts['lr_decay']),
        'lambda': optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts['lr_decay'] ** epoch),
        'const': optim.lr_scheduler.ConstantLR(optimizer, opts['lr_decay'], opts['lrs_total_steps']),
        'poly': optim.lr_scheduler.PolynomialLR(optimizer, opts['lrs_total_steps'], opts['lr_decay']),
        'multistep': optim.lr_scheduler.MultiStepLR(optimizer, opts['lrs_milestones'], opts['lr_decay']),
        'cosan': optim.lr_scheduler.CosineAnnealingLR(optimizer, opts['lrs_total_steps'], opts['lr_min_value']),
        'linear': optim.lr_scheduler.LinearLR(optimizer, opts['lr_min_decay'], opts['lr_decay'], opts['lrs_total_steps']),
        'cosanwr': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, opts['lrs_restart_steps'], opts['lrs_rfactor'], opts['lr_min_value']),
        'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, opts['lrs_mode'], opts['lrs_dfactor'], opts['lrs_patience'], opts['lrs_thresh'], 
                                                        opts['lrs_thresh_mode'], opts['lrs_cooldown'], opts['lr_min_value'], opts['lr_min_decay']),
    }.get(opts['lr_scheduler'], None)
    assert optimizer is not None, "Unknown learning rate scheduler: {}".format(opts['lr_scheduler'])
    return optimizer, lr_scheduler