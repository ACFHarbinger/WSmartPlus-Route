"""Unit tests for tasks/losses.py."""

import torch
import torch.nn.functional as F
import pytest
from logic.src.utils.tasks import losses

def test_problem_symmetricity_loss_zeros():
    """Test symmetry loss with identical rewards (zero advantage)."""
    # [Batch, Augmentations]
    rewards = torch.tensor([[10.0, 10.0], [5.0, 5.0]])
    log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])

    # Mean reward across dim 1 is same as reward -> advantage is 0 -> loss is 0
    loss = losses.problem_symmetricity_loss(rewards, log_probs, dim=1)
    assert torch.isclose(loss, torch.tensor(0.0))

def test_problem_symmetricity_loss_basic():
    """Test basic arithmetic of symmetry loss."""
    # Batch=1, Aug=2
    # R = [10, 20], Mean=15
    # Adv = [-5, 5]
    # LogP = [-1, -2] (dummy)
    # Loss = - (Adv * LogP) = - ( [-5*-1, 5*-2] ) = - ( [5, -10] ) = [-5, 10]
    # Mean Loss = 2.5

    rewards = torch.tensor([[10.0, 20.0]])
    log_probs = torch.tensor([[-1.0, -2.0]])

    loss = losses.problem_symmetricity_loss(rewards, log_probs, dim=1)
    assert torch.isclose(loss, torch.tensor(2.5))

def test_solution_symmetricity_loss_dim():
    """Test solution symmetry loss on last dimension."""
    # Same math as above but dim=-1
    rewards = torch.tensor([[10.0, 20.0]])
    log_probs = torch.tensor([[-1.0, -2.0]])

    loss = losses.solution_symmetricity_loss(rewards, log_probs, dim=-1)
    assert torch.isclose(loss, torch.tensor(2.5))

def test_invariance_loss():
    """Test invariance loss (cosine similarity)."""
    # Batch=1, Aug=2, Dim=2
    # Embeddings identical -> Cosine Sim = 1.0
    # Expected Loss: -(1.0 / (2-1)) = -1.0

    bs = 1
    aug = 2
    dim = 2

    # 3D input: [Batch*Aug, Graph, Dim]
    embed = torch.ones((bs * aug, 1, dim))

    loss = losses.invariance_loss(embed, num_augment=aug)
    assert torch.isclose(loss, torch.tensor(-1.0))

def test_invariance_loss_orthogonal():
    """Test invariance loss with orthogonal vectors."""
    # Aug 1: [1, 0]
    # Aug 2: [0, 1]
    # Sim = 0
    # Loss = 0

    embed = torch.tensor([
        [[1.0, 0.0]],
        [[0.0, 1.0]]
    ]) # Shape [2, 1, 2] -> Batch*Aug=2

    loss = losses.invariance_loss(embed, num_augment=2)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
