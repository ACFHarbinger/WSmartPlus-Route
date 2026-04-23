"""NARGNN Policy: Non-Autoregressive GNN Policy.

This module provides `NARGNNPolicy`, which predicts an edge-wise heatmap using a
deep GNN and then applies a non-autoregressive decoder (e.g., greedy or beam
search) to extract solution sequences.

Attributes:
    NARGNNPolicy: Heatmap-based construction policy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs import get_env
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.decoder import (
    NonAutoregressiveDecoder,
)
from logic.src.models.common.non_autoregressive.encoder import (
    NonAutoregressiveEncoder,
)
from logic.src.models.common.non_autoregressive.policy import (
    NonAutoregressivePolicy,
)
from logic.src.models.subnets.decoders.nar import SimpleNARDecoder
from logic.src.models.subnets.encoders.nargnn import NARGNNEncoder


class NARGNNPolicy(NonAutoregressivePolicy):
    """Base Non-autoregressive policy for NCO.

    Produces a heatmap of size [N, N] for N nodes, where each entry represents
    the predicted probability of an edge being part of the optimal solution.

    Attributes:
        train_strategy (str): Decoding mode for training.
        val_strategy (str): Decoding mode for validation.
        test_strategy (str): Decoding mode for testing.
        encoder (NARGNNEncoder): Deep GNN for edge-prob encoding.
        decoder (NonAutoregressiveDecoder): Builder that converts heatmaps to routes.
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
        train_strategy: str = "sampling",
        val_strategy: str = "greedy",
        test_strategy: str = "greedy",
        **constructive_policy_kw: Any,
    ) -> None:
        """Initializes the NARGNNPolicy.

        Args:
            encoder: instance of the GNN heatmap predictor.
            decoder: instance of the NAR sequence builder.
            embed_dim: size of latent node/edge features.
            env_name: task identifier.
            init_embedding: projection module for raw nodes.
            edge_embedding: projection module for raw edges.
            graph_network: core GNN message-passing module.
            heatmap_generator: final MLP that predicts edge scores.
            num_layers_heatmap_generator: depth of the output MLP.
            num_layers_graph_encoder: depth of the GNN stack.
            act_fn: activation function for GNN layers.
            agg_fn: neighbor pooling strategy ('mean', 'sum').
            linear_bias: toggle bias in linear projections.
            train_strategy: selection mode during learning.
            val_strategy: selection mode during validation.
            test_strategy: selection mode during testing.
            **constructive_policy_kw: extra params for the base NAR policy.
        """
        if encoder is None:
            # NARGNNEncoder implements functional interface of NonAutoregressiveEncoder
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

        if decoder is None:
            decoder = SimpleNARDecoder()

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            **constructive_policy_kw,
        )

        self.train_strategy = train_strategy
        self.val_strategy = val_strategy
        self.test_strategy = test_strategy

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        num_starts: int = 1,
        phase: str = "test",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculates heatmaps and extracts solution sequences.

        Args:
            td: problem state container.
            env: problem dynamics.
            num_starts: parallel starts for multi-path sampling.
            phase: Current mode ('train', 'val', 'test').
            **kwargs: extra params for decoding and strategy selection.

        Returns:
            Dict[str, Any]: Result map including:
                - actions: node sequences [B, num_starts, SeqLen].
                - reward: Tensor of environment rewards.
                - log_likelihood: Cumulative path probabilities.
                - heatmap: Raw edge inclusion scores [B, N, N].
                - node_embed: final contextual node features.
        """
        # 1. Edge-wise probability heatmap prediction
        heatmap, node_embed = self.encoder(td)  # type: ignore[misc]

        # 2. Strategy selection based on phase
        strategy = kwargs.get("strategy")
        if strategy is None:
            strategy = getattr(self, f"{phase}_strategy", "greedy")

        if env is None:
            env = get_env(self.env_name)  # type: ignore[arg-type]

        # 3. Constructive search using non-autoregressive decoding
        logprobs, actions, td, env = self.common_decoding(
            strategy=strategy,  # type: ignore[arg-type]
            td=td,
            env=env,
            heatmap=heatmap,
            num_starts=num_starts,
            **kwargs,
        )

        # 4. Result preparation
        from logic.src.utils.decoding import get_log_likelihood

        out = {
            "actions": actions,
            "reward": env.get_reward(td, actions) if env is not None else None,
            "log_likelihood": get_log_likelihood(logprobs, None, td.get("mask", None), True),
            "heatmap": heatmap,
            "node_embed": node_embed,
        }
        return out
