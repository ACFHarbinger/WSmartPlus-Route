
import pytest
import torch
from torch import nn
from logic.src.models.subnets.modules.mpnn import MessagePassingLayer, MPNNEncoder

class TestMPNN:
    @pytest.fixture
    def setup_data(self):
        batch_size = 2
        num_nodes = 5
        node_dim = 16
        edge_dim = 16

        # Random node features
        x = torch.randn(batch_size * num_nodes, node_dim)

        # Dense edge index for fully connected graph
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(batch_size * num_nodes), batch_size * num_nodes),
            torch.tile(torch.arange(batch_size * num_nodes), (batch_size * num_nodes,))
        ], dim=0)

        # Filter to keep only edges within the same batch graph
        # This simplifies the test setup to simulating a batch of graphs
        # For simplicity, let's just test single large graph or use correct batch indexing
        # Correct approach:
        edge_list = []
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_list.append([b * num_nodes + i, b * num_nodes + j])
        edge_index = torch.tensor(edge_list).t()

        edge_attr = torch.randn(edge_index.size(1), edge_dim)

        return x, edge_index, edge_attr

    def test_message_passing_layer_forward(self, setup_data):
        x, edge_index, edge_attr = setup_data
        node_dim = x.size(1)
        edge_dim = edge_attr.size(1)

        layer = MessagePassingLayer(node_dim, edge_dim, hidden_dim=32)

        out_x, out_edge_attr = layer(x, edge_index, edge_attr)

        assert out_x.shape == x.shape
        assert out_edge_attr.shape == edge_attr.shape
        assert not torch.isnan(out_x).any()
        assert not torch.isnan(out_edge_attr).any()

    def test_mpnn_encoder_forward(self, setup_data):
        x, edge_index, edge_attr = setup_data
        node_dim = x.size(1)
        edge_dim = edge_attr.size(1)

        encoder = MPNNEncoder(
            num_layers=2,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=32
        )

        # If hidden_dim != node_dim, output shape changes
        out_x, out_edge_attr = encoder(x, edge_index, edge_attr)

        assert out_x.shape == (x.size(0), 32)
        assert out_edge_attr.shape == (edge_attr.size(0), 32)
        assert not torch.isnan(out_x).any()

    def test_mpnn_input_output_dims(self):
        # Test projection logic
        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.randn(2, 4)

        encoder = MPNNEncoder(
            num_layers=1,
            node_dim=8,
            edge_dim=4,
            hidden_dim=16
        )

        out_x, out_edge = encoder(x, edge_index, edge_attr)
        assert out_x.shape == (10, 16)
        assert out_edge.shape == (2, 16)
