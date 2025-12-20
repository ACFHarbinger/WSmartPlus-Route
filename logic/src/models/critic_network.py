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

        self.is_wc = problem.NAME == 'wcvrp' or problem.NAME == 'cwcvrp' or problem.NAME == 'sdwcvrp'
        self.is_vrpp = problem.NAME == 'vrpp' or problem.NAME == 'cvrpp'

        assert self.is_wc or self.is_vrpp, "Unsupported problem: {}".format(problem.NAME)

        # Problem specific context parameters
        node_dim = 3  # x, y, demand / prize / waste -- vrpp has waste, wc has waste.
        
        # Special embedding projection for depot node
        self.init_embed_depot = nn.Linear(2, embedding_dim)

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
        if self.is_vrpp or self.is_wc:
            features = ('waste',)
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
        assert False, "Unsupported problem"

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