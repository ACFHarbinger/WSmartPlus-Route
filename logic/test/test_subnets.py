import torch
import torch.nn as nn

from logic.src.models.subnets.gac_encoder import GraphAttConvEncoder
from logic.src.models.subnets.gat_decoder import GraphAttentionDecoder
from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder
from logic.src.models.subnets.gcn_encoder import GraphConvolutionEncoder
from logic.src.models.subnets.grf_predictor import GatedRecurrentFillPredictor
from logic.src.models.subnets.mlp_encoder import MLPEncoder
from logic.src.models.subnets.ptr_decoder import PointerDecoder
from logic.src.models.subnets.ptr_encoder import PointerEncoder
from logic.src.models.subnets.tgc_encoder import TransGraphConvEncoder


class TestGACEncoder:
    def test_init(self):
        model = GraphAttConvEncoder(n_heads=2, embed_dim=16, n_layers=1, n_groups=4)
        assert isinstance(model, nn.Module)

    def test_forward(self):
        embed_dim = 16
        model = GraphAttConvEncoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)
        edges = torch.randn(batch, graph_size, graph_size) 
        
        output = model(x, edges)
        assert output.shape == (batch, graph_size, embed_dim)

class TestGATDecoder:
    def test_forward(self):
        embed_dim = 16
        model = GraphAttentionDecoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_groups=4)
        batch = 2
        graph_size = 5
        
        # MHA expects q to have sequence dimension if it asserts dim(2)
        q = torch.randn(batch, embed_dim).unsqueeze(1) # (batch, 1, embed_dim)
        h = torch.randn(batch, graph_size, embed_dim) 
        mask = torch.zeros(batch, graph_size, dtype=torch.bool)
        
        output = model(q, h, mask)
        # Returns (batch, 1, 2) since q seq_len is 1
        # Wait, gat_decoder returns softmax(out).
        # out = self.projection(self.dropout(h)). 
        # h follows q shape.
        assert output.shape == (batch, 1, 2)

class TestGATEncoder:
    def test_forward(self):
        embed_dim = 16
        model = GraphAttentionEncoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)
        
        output = model(x)
        assert output.shape == (batch, graph_size, embed_dim)

class TestGCNEncoder:
    def test_forward(self):
        hidden = 16
        # Fixed Normalization to handle default n_groups or pass n_groups if we can.
        # GCNEncoder uses default GraphConvolution.
        # I fixed Normalization class to default n_groups=1 if None.
        model = GraphConvolutionEncoder(n_layers=1, hidden_dim=hidden, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, hidden)
        
        # GCN expects edges to be Long (indices) for Embedding?
        # Source: self.init_embed_edges(edges.type(torch.long))
        edges = torch.randint(0, 2, (batch, graph_size, graph_size)) 
        
        output = model(x, edges)
        assert output.shape == (batch, graph_size, hidden)

class TestGatedRecurrentFillPredictor:
    def test_forward(self):
        hidden = 16
        model = GatedRecurrentFillPredictor(input_dim=1, hidden_dim=hidden, num_layers=1, activation='relu')
        batch = 2
        seq = 5
        x = torch.randn(batch, seq, 1)
        
        output = model(x)
        assert output.shape == (batch, 1)

class TestMLPEncoder:
    def test_forward(self):
        dim = 16
        model = MLPEncoder(n_layers=1, hidden_dim=dim)
        batch = 2
        nodes = 5
        x = torch.randn(batch, nodes, dim)
        
        output = model(x)
        assert output.shape == (batch, nodes, dim)

class TestPointerDecoder:
    def test_forward(self):
        embed_dim = 16
        hidden_dim = 16
        model = PointerDecoder(
            embedding_dim=embed_dim, 
            hidden_dim=hidden_dim, 
            tanh_exploration=10, 
            use_tanh=True
        )
        
        batch = 2
        seq = 5
        decoder_input = torch.randn(batch, embed_dim)
        embedded_inputs = torch.randn(seq, batch, embed_dim)
        hidden = (torch.randn(batch, hidden_dim), torch.randn(batch, hidden_dim))
        context = torch.randn(seq, batch, hidden_dim)
        
        model.decode_type = "greedy"
        
        (log_p, selections), hidden_out = model(decoder_input, embedded_inputs, hidden, context)
        
        assert log_p.shape == (batch, seq, seq)
        assert selections.shape == (batch, seq)

class TestPointerEncoder:
    def test_forward(self):
        input_dim = 10
        hidden_dim = 16
        model = PointerEncoder(input_dim, hidden_dim)
        
        seq = 5
        batch = 2
        x = torch.randn(seq, batch, input_dim) 
        hidden = model.init_hidden(hidden_dim)
        
        # Expand hidden
        h0 = hidden[0].unsqueeze(0).unsqueeze(1).expand(1, batch, hidden_dim).contiguous()
        c0 = hidden[1].unsqueeze(0).unsqueeze(1).expand(1, batch, hidden_dim).contiguous()
        
        out, _ = model(x, (h0, c0))
        assert out.shape == (seq, batch, hidden_dim)

class TestTGCEncoder:
    def test_forward(self):
        embed_dim = 16
        # n_groups=4 compatible with 16
        model = TransGraphConvEncoder(n_heads=2, embed_dim=embed_dim, n_layers=1, n_sublayers=1, n_groups=4)
        batch = 2
        graph_size = 5
        x = torch.randn(batch, graph_size, embed_dim)
        # TGC expects float edges (adj matrix) as it uses GraphConvolution with 'mean' agg?
        # Yes, line 100 agg='mean'.
        # I fixed GraphConvolution to support batched masks.
        edges = torch.randn(batch, graph_size, graph_size)
        
        output = model(x, edges)
        assert output.shape == (batch, graph_size, embed_dim)
