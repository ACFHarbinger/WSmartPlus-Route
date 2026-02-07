"""
Backward-compatibility shim for CriticNetwork.

The canonical CriticNetwork implementation is now in
``logic.src.models.policies.critic``. This module re-exports both
the new (TensorDict-based) and legacy (problem/factory-based) versions.

**New code should import from** ``logic.src.models.policies.critic``.
"""

import warnings

import torch.nn as nn

# Re-export the canonical CriticNetwork and factory
from logic.src.models.policies.critic import CriticNetwork, create_critic_from_actor  # noqa: F401
from logic.src.models.subnets.embeddings.context_embedder import VRPPContextEmbedder, WCContextEmbedder
from logic.src.models.subnets.modules import ActivationFunction


class LegacyCriticNetwork(nn.Module):
    """
    Legacy Critic Network using problem objects and component factories.

    This is the original implementation kept for backward compatibility with
    the legacy training pipeline. New code should use
    ``logic.src.models.policies.critic.CriticNetwork`` instead.
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
        super().__init__()
        warnings.warn(
            "LegacyCriticNetwork is deprecated. Use logic.src.models.policies.critic.CriticNetwork instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.aggregation_graph = aggregation_graph

        self.is_wc = problem.NAME in ("wcvrp", "cwcvrp", "sdwcvrp")
        self.is_vrpp = problem.NAME in ("vrpp", "cvrpp")

        assert self.is_wc or self.is_vrpp, f"Unsupported problem: {problem.NAME}"

        node_dim = 3

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
        edges = inputs.get("edges", None)
        embeddings = self.encoder(self._init_embed(inputs), edges)
        if self.aggregation_graph == "avg":
            graph_embeddings = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embeddings = embeddings.sum(1)
        else:
            assert self.aggregation_graph == "max", f"Unsupported aggregation: {self.aggregation_graph}"
            graph_embeddings = embeddings.max(1)[0]
        return self.value_head(graph_embeddings)


__all__ = [
    "CriticNetwork",
    "LegacyCriticNetwork",
    "create_critic_from_actor",
]
