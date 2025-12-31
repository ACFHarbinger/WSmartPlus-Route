from logic.src.pipeline.reinforcement_learning.core.trainers import (
    StandardTrainer, RWATrainer, ContextualBanditTrainer, 
    TDLTrainer, MORLTrainer, HyperNetworkTrainer, HRLTrainer, 
    TimeTrainer
)


def train_reinforce_epoch(model, optimizer, baseline, lr_scheduler, scaler, epoch, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Wrapper for running a single epoch of standard training.
    """
    trainer = StandardTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.day = epoch # Set current epoch context
    trainer.update_context() # Prepare dataset
    trainer.train_day() # Run training
    trainer.post_day_processing()
    return None # Return value not strictly used in loop usually


def train_reinforce_over_time_rwa(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    trainer = RWATrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_reinforce_over_time_cb(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    trainer = ContextualBanditTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_reinforce_over_time_tdl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    trainer = TDLTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_reinforce_over_time_morl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    trainer = MORLTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_over_time_with_hypernetwork(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    trainer = HyperNetworkTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.hyper_optimizer


def train_reinforce_over_time_hrl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    trainer = HRLTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.hrl_manager


def train_reinforce_over_time(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    if opts.get('rwa', False):
        return train_reinforce_over_time_rwa(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('cb', False):
        return train_reinforce_over_time_cb(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('tdl', False):
        return train_reinforce_over_time_tdl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('morl', False):
        return train_reinforce_over_time_morl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('hypernetwork', False):
        return train_over_time_with_hypernetwork(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('hrl', False):
        return train_reinforce_over_time_hrl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    else:
        # Default simple time training (Standard Time ?)
        # Use TimeTrainer
        trainer = TimeTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        trainer.train()
        return model, None
