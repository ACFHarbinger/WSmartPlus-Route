import numpy as np


def get_linear_cost_weight(epoch, start_epoch, total_epochs, cw_start, cw_end):
    """
    Linearly increase cost_weight from cw_start to cw_end over training.
    
    Args:
        epoch: Current epoch
        start_epoch: Starting epoch (your START variable)
        total_epochs: Total number of epochs (your TOTAL_EPOCHS)
        cw_start: Initial cost_weight (e.g., 0.2)
        cw_end: Final cost_weight (e.g., 2.0)
    """
    if epoch < start_epoch:
        return cw_start
    
    progress = (epoch - start_epoch) / (total_epochs - start_epoch)
    current_cw = cw_start + (cw_end - cw_start) * progress
    return current_cw


def get_staged_cost_weight(epoch, start_epoch, total_epochs):
    """
    Increase cost_weight in stages to allow model to adapt.
    """
    progress = (epoch - start_epoch) / (total_epochs - start_epoch)
    
    if progress < 0.3:  # First 30% of training
        return 0.2
    elif progress < 0.6:  # Next 30%
        return 0.5
    elif progress < 0.8:  # Next 20%
        return 1.0
    else:  # Final 20%
        return 2.0


def get_exponential_cost_weight(epoch, start_epoch, total_epochs, cw_start, cw_end, k=5):
    """
    Exponentially increase cost_weight (slow start, fast end).
    
    Args:
        k: Steepness parameter (higher = more aggressive curve)
    """
    if epoch < start_epoch:
        return cw_start
    
    progress = (epoch - start_epoch) / (total_epochs - start_epoch)
    
    # Exponential curve: y = start + (end - start) * (e^(kx) - 1) / (e^k - 1)
    normalized = (np.exp(k * progress) - 1) / (np.exp(k) - 1)
    current_cw = cw_start + (cw_end - cw_start) * normalized
    return current_cw
