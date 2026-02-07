"""
This module contains the Critic Network implementation for RL baselines.
"""

import torch.nn as nn

from .context_embedder import VRPPContextEmbedder, WCContextEmbedder
from .modules import ActivationFunction


# Attention, Learn to Solve Routing Problems
class CriticNetwork(nn.Module):
    """
    Critic Network for estimating the value of a problem state in Reinforcement Learning.

    Used as a baseline to reduce variance in gradient estimation (e.g., in PPO/REINFORCE).
    """

    def __init__(
        self,
        problem,
        component_factory,
        embed_dim,
        hidden_dim,
        n_layers,
        n_sublayers,
        encoder_normalization="batch",
        activation="gelu",
        n_heads=8,
        aggregation_graph="avg",
        dropout_rate=0.0,
        temporal_horizon=0,
    ):
        """
        Initialize the Critic Network.

        Args:
            problem (object): The problem instance wrapper.
            component_factory (NeuralComponentFactory): Factory to create sub-components.
            embed_dim (int): Dimension of the embedding vectors.
            hidden_dim (int): Dimension of the hidden layers.
            n_layers (int): Number of encoder layers.
            n_sublayers (int): Number of sub-layers in encoder.
            encoder_normalization (str, optional): Normalization type for encoder. Defaults to 'batch'.
            activation (str, optional): Activation function name. Defaults to 'gelu'.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            aggregation_graph (str, optional): Graph aggregation method ('avg', 'sum', 'max'). Defaults to "avg".
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            temporal_horizon (int, optional): Horizon for temporal features. Defaults to 0.
        """
        super(CriticNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.aggregation_graph = aggregation_graph

        self.is_wc = problem.NAME == "wcvrp" or problem.NAME == "cwcvrp" or problem.NAME == "sdwcvrp"
        self.is_vrpp = problem.NAME == "vrpp" or problem.NAME == "cvrpp"

        assert self.is_wc or self.is_vrpp, "Unsupported problem: {}".format(problem.NAME)

        # Problem specific context parameters
        node_dim = 3  # x, y, demand / prize / waste -- vrpp has waste, wc has waste.

        if self.is_wc:
            self.context_embedder = WCContextEmbedder(embed_dim, node_dim=node_dim, temporal_horizon=temporal_horizon)
        else:
            self.context_embedder = VRPPContextEmbedder(embed_dim, node_dim=node_dim, temporal_horizon=temporal_horizon)

        self.encoder = component_factory.create_encoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_sublayers=n_sublayers,
            normalization=encoder_normalization,
            activation=activation,
            dropout_rate=dropout_rate,
            feed_forward_hidden=self.hidden_dim,
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, 1),
        )

    def _init_embed(self, nodes):
        return self.context_embedder.init_node_embeddings(nodes)

    def forward(self, inputs):
        """
        Forward pass of the Critic Network.

        Args:
            inputs (dict): The input data dictionary containing problem state (nodes, edges, etc.).

        Returns:
            torch.Tensor: The estimated value of the current state.
        """
        edges = inputs.get("edges", None)
        embeddings = self.encoder(self._init_embed(inputs), edges)
        if self.aggregation_graph == "avg":
            graph_embeddings = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embeddings = embeddings.sum(1)
        else:
            assert self.aggregation_graph == "max", "Unsupported graph aggregation method: {}".format(
                self.aggregation_graph
            )
            graph_embeddings = embeddings.max(1)[0]
        return self.value_head(graph_embeddings)
