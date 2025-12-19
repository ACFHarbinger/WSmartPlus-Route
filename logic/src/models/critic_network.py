import torch
import torch.nn as nn

from .modules import ActivationFunction
from .subnets import PointerEncoder, PointerAttention


# Attention, Learn to Solve Routing Problems
class CriticNetwork(nn.Module):
    def __init__(
        self,
        problem,
        encoder_class,
        embedding_dim,
        hidden_dim,
        n_layers,
        n_sublayers,
        encoder_normalization='batch',
        activation='gelu',
        n_heads=8,
        aggregation_graph="avg",
        dropout_rate=0.,
    ):
        super(CriticNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.encoder_class = encoder_class
        self.aggregation_graph = aggregation_graph

        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_wc = problem.NAME == 'wcvrp' or problem.NAME == 'cwcvrp' or problem.NAME == 'sdwcvrp'
        self.is_vrpp = problem.NAME == 'vrpp' or problem.NAME == 'cvrpp'

         # Problem specific context parameters
        if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc or self.is_vrpp:
            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize / waste

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
        else:
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            node_dim = 2

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.encoder = self.encoder_class(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_layers,
            n_sublayers=n_sublayers,
            normalization=encoder_normalization,
            activation=activation,
            dropout_rate=dropout_rate,
            feed_forward_hidden=self.hidden_dim
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, 1)
        )
    
    def _init_embed(self, nodes):
        if self.is_vrpp or self.is_vrp or self.is_wc or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand',)  # [batch_size, graph_size]
            elif self.is_vrpp or self.is_wc:
                features = ('waste',)
            elif self.is_orienteering:
                features = ('prize',)
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(nodes['depot'])[:, None, :],
                    self.init_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        nodes['loc'],  # [batch_size, graph_size, 2]
                        *(nodes[feat][:, :, None] for feat in features)  # [batch_size, graph_size]
                    ), -1))  # [batch_size, graph_size, node_dim]
                ),
                1
            )
        # TSP
        return self.init_embed(nodes['loc'])

    def forward(self, inputs):
        """
        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        edges = inputs.pop('edges')
        embeddings = self.encoder(self._init_embed(inputs), edges)
        if self.aggregation_graph == "avg":
            graph_embeddings = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embeddings = embeddings.sum(1)
        else:
            assert self.aggregation_graph == "max", "Unsupported graph aggregation method: {}".format(self.aggregation_graph)
            graph_embeddings = embeddings.max(1)[0]
        return self.value_head(graph_embeddings)
    

class CriticNetworkLSTM(nn.Module):
    """Useful as a baseline in REINFORCE updates"""
    def __init__(self, embedding_dim, hidden_dim, n_process_block_iters, tanh_exploration, use_tanh):
        super(CriticNetworkLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters
        self.encoder = PointerEncoder(embedding_dim, hidden_dim)        
        self.process_block = PointerAttention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous()
        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out