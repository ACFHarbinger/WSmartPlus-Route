"""
NARGNN Policy.

Non-autoregressive GNN-based policy for combinatorial optimization.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from logic.src.models.policies.common.nonautoregressive import (
    NonAutoregressiveDecoder,
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from logic.src.models.subnets.encoders.nargnn_encoder import NARGNNEncoder


class NARGNNPolicy(NonAutoregressivePolicy):
    """
    Base Non-autoregressive policy for NCO construction methods.

    Creates a heatmap of NxN for N nodes (i.e., heuristic) that models
    the probability to go from one node to another for all nodes.

    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (non-autoregressively) to construct the solution

    Warning:
        The effectiveness of the non-autoregressive approach can vary
        significantly across different problem types and configurations.
        It may require careful tuning of the model architecture and
        decoding strategy to achieve competitive results.

    Args:
        encoder: Encoder module. Can be passed by sub-classes.
        decoder: Decoder module. Defaults to non-autoregressive decoder.
        embed_dim: Dimension of the embeddings.
        env_name: Name of the environment used to initialize embeddings.
        init_embedding: Model to use for the initial embedding. If None, use default.
        edge_embedding: Model to use for the edge embedding. If None, use default.
        graph_network: Model to use for the graph network. If None, use default.
        heatmap_generator: Model to use for the heatmap generator. If None, use default.
        num_layers_heatmap_generator: Number of layers in the heatmap generator.
        num_layers_graph_encoder: Number of layers in the graph encoder.
        act_fn: Activation function to use in the encoder.
        agg_fn: Aggregation function to use in the encoder.
        linear_bias: Whether to use bias in the encoder.
        train_decode_type: Type of decoding during training.
        val_decode_type: Type of decoding during validation.
        test_decode_type: Type of decoding during testing.
        **constructive_policy_kw: Additional keyword arguments.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        embed_dim: int = 64,
        env_name: str = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        act_fn: str = "silu",
        agg_fn: str = "mean",
        linear_bias: bool = True,
        train_decode_type: str = "multistart_sampling",
        val_decode_type: str = "multistart_greedy",
        test_decode_type: str = "multistart_greedy",
        **constructive_policy_kw,
    ) -> None:
        if encoder is None:
            # NARGNNEncoder doesn't inherit from NonAutoregressiveEncoder but has compatible interface
            encoder = NARGNNEncoder(  # type: ignore[assignment]
                embed_dim=embed_dim,
                env_name=env_name,
                init_embedding=init_embedding,
                edge_embedding=edge_embedding,
                graph_network=graph_network,
                heatmap_generator=heatmap_generator,
                num_layers_heatmap_generator=num_layers_heatmap_generator,
                num_layers_graph_encoder=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
                linear_bias=linear_bias,
            )

        # NARGNN doesn't use a separate decoder - solutions are constructed directly from heatmaps
        # Decoder is None and handled by parent class NonAutoregressivePolicy

        # Pass to constructive policy
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )
