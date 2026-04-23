r"""PolyNet Policy: Multi-Strategy Autoregressive Policy.

This module provides `PolyNetPolicy`, which conditions a standard transformer
encoder-decoder architecture on a strategy index $k \in \{1, \dots, K\}$. This
allows the model to represent a population of distinct routing heuristics within
a single weight set.

Attributes:
    PolyNetPolicy: Strategy-conditioned constructive policy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.subnets.decoders.polynet import PolyNetDecoder
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.gat.encoder import GraphAttentionEncoder


class PolyNetPolicy(AutoregressivePolicy):
    """PolyNet Policy for population-based search.

    Utilizes static binary vectors injected into the attention mechanism to
    branch the decoding path. This creates $K$ specialized "experts" that share
    the same encoder but develop unique searching behaviors.

    Attributes:
        k (int): Number of learnable behavior branches.
        embed_dim (int): feature width.
        temperature (float): softmax temperature.
        tanh_clipping (float): logit clipping range.
        mask_logits (bool): toggle invalid action masking.
        init_embedding (nn.Module): Problem-specific initial projector.
        train_strategy (str): Training construction mode.
        val_strategy (str): Validation construction mode.
        test_strategy (str): Testing construction mode.
    """

    def __init__(
        self,
        k: int,
        encoder: Optional[nn.Module] = None,
        encoder_type: str = "AM",
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "vrpp",
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_strategy: str = "sampling",
        val_strategy: str = "sampling",
        test_strategy: str = "sampling",
        **kwargs: Any,
    ) -> None:
        """Initializes the PolyNet policy.

        Args:
            k: number of distinct behavior strategies.
            encoder: Backbone graph processor.
            encoder_type: identity of the architecture ('AM', 'MatNet').
            embed_dim: total latent feature width.
            num_encoder_layers: depth of the attention stack.
            num_heads: attention heads count.
            normalization: module scaling type.
            feedforward_hidden: MLP expansion width.
            env_name: target task identifier.
            temperature: Softmax scaling.
            tanh_clipping: value range for logit squashing.
            mask_logits: toggle for valid action enforcement.
            train_strategy: default search mode for training.
            val_strategy: default search mode for validation.
            test_strategy: default search mode for testing.
            **kwargs: extra params for PolyNetDecoder.
        """
        if encoder is None:
            encoder = GraphAttentionEncoder(
                n_heads=num_heads,
                embed_dim=embed_dim,
                n_layers=num_encoder_layers,
                normalization=normalization,
                feed_forward_hidden=feedforward_hidden,
            )

        decoder = PolyNetDecoder(
            k=k,
            encoder_type=encoder_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            **kwargs,
        )

        super().__init__(
            env_name=env_name,
            encoder=encoder,
            decoder=decoder,
        )

        self.k = k
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits

        self.init_embedding = get_init_embedding(env_name, embed_dim)

        self.train_strategy = train_strategy
        self.val_strategy = val_strategy
        self.test_strategy = test_strategy

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = True,
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Performs strategy-conditioned constructive search.

        Args:
            td: current state container.
            env: environment dynamics (uses standard if None).
            phase: pipeline state ('train', 'val', 'test').
            return_actions: toggle for explicit action sequence in output.
            num_starts: number of parallel strategy constructors.
            **kwargs: extra decoding configurations.

        Returns:
            Dict[str, Any]: results mapping rewards, log_probs, and actions.
        """
        # 1. Project raw inputs (e.g. coordinates to latent vectors)
        embedding = self.init_embedding(td)

        # 2. Contextualize nodes via Graph Attention
        embeddings = self.encoder(embedding)  # type: ignore[attr-defined, misc]

        # 3. Strategy strategy selection
        strategy = kwargs.pop("strategy", None)
        if strategy is None:
            strategy = getattr(self, f"{phase}_strategy")

        # 4. Strategy-conditioned recurrent construction
        log_likelihood, actions = self.decoder(  # type: ignore[attr-defined, misc]
            td,
            embeddings,
            env,
            strategy=strategy,
            num_starts=num_starts,
            **kwargs,
        )

        # 5. Reward extraction
        from logic.src.envs import get_env

        if env is None:
            env = get_env(self.env_name)  # type: ignore[arg-type]
        reward = env.get_reward(td, actions)  # type: ignore[attr-defined, union-attr]

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": actions if return_actions else None,
        }
