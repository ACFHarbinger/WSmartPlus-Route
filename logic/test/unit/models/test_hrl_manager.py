import torch
import pytest
from logic.src.models.hrl_manager.model import GATLSTManager

def test_gatlst_manager_init():
    manager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=10,
        global_input_dim=2,
        hidden_dim=16,
        lstm_hidden=8,
        num_layers_gat=2,
        n_heads=2,
        device="cpu"
    )
    assert isinstance(manager, GATLSTManager)
    assert manager.hidden_dim == 16

def test_gatlst_manager_forward():
    batch_size = 4
    num_nodes = 10
    history_len = 10

    manager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=history_len,
        global_input_dim=2,
        hidden_dim=16,
        lstm_hidden=8,
        num_layers_gat=2,
        n_heads=2,
        device="cpu"
    )

    static = torch.randn(batch_size, num_nodes, 2)
    dynamic = torch.randn(batch_size, num_nodes, history_len)
    global_features = torch.randn(batch_size, 2)

    must_go_logits, gate_logits, value = manager(static, dynamic, global_features)

    assert must_go_logits.shape == (batch_size, num_nodes, 2)
    assert gate_logits.shape == (batch_size, 2)
    assert value.shape == (batch_size, 1)

def test_gatlst_manager_select_action():
    batch_size = 4
    num_nodes = 10
    history_len = 10

    manager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=history_len,
        global_input_dim=2,
        hidden_dim=16,
        lstm_hidden=8,
        num_layers_gat=2,
        n_heads=2,
        device="cpu"
    )

    static = torch.randn(batch_size, num_nodes, 2)
    dynamic = torch.randn(batch_size, num_nodes, history_len)
    global_features = torch.randn(batch_size, 2)

    must_go_action, gate_action, value = manager.select_action(
        static, dynamic, global_features, deterministic=False
    )

    assert must_go_action.shape == (batch_size, num_nodes)
    assert gate_action.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
