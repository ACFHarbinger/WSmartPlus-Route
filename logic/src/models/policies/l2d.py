"""
L2D (Learning to Dispatch) Policy for Job Shop Scheduling.

Based on "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning" (Zhang et al. 2020).
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict

from logic.src.models.policies.common.constructive import ConstructivePolicy
from logic.src.models.subnets.decoders.l2d import L2DDecoder
from logic.src.models.subnets.encoders.l2d_encoder import L2DEncoder


class L2DPolicy(ConstructivePolicy):
    """
    Learning to Dispatch (L2D) Policy for JSSP.

    Uses L2DEncoder to encode the disjunctive graph state and selects
    the next job to schedule an operation for.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        feedforward_hidden: int = 512,
        env_name: str = "jssp",
        temp: float = 1.0,
        tanh_clipping: float = 10.0,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **kwargs,
    ):
        """
        Initialize L2DPolicy.

        Args:
            embed_dim: Embedding dimension.
            num_encoder_layers: Number of GNN layers in encoder.
            feedforward_hidden: Hidden dimension in GNN feedforward.
            env_name: Environment name (default: "jssp").
            temp: Temperature for sampling.
            tanh_clipping: Tanh clipping value.
            train_decode_type: Decode type during training.
            val_decode_type: Decode type during validation.
            test_decode_type: Decode type during testing.
            **kwargs: Additional arguments for ConstructivePolicy.
        """
        super().__init__(
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **kwargs,
        )

        self.encoder = L2DEncoder(
            embed_dim=embed_dim, num_layers=num_encoder_layers, feedforward_hidden=feedforward_hidden
        )

        self.decoder = L2DDecoder(embed_dim=embed_dim, temp=temp, tanh_clipping=tanh_clipping)

    def forward(
        self,
        td: TensorDict,
        env: Optional[Any] = None,
        decode_type: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> dict:
        """
        Forward pass (Autoregressive Rollout).
        """
        if env is None:
            raise ValueError("L2DPolicy requires 'env' to play out the episode.")

        device = td.device
        batch_size = td.batch_size

        # Reset env
        td = env.reset(td)

        # Buffers
        actions = []
        log_probs = []

        while not td["done"].all():
            # Encode
            embeddings = self.encoder(td)

            # Decode
            log_prob, action = self.decoder(td, embeddings, env, decode_type=decode_type)

            # Step
            td["action"] = action
            td = env.step(td)["next"]

            # Store
            actions.append(action)
            log_probs.append(log_prob)

        # Stack
        out = {
            "reward": td["reward"].squeeze(-1),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1)
            if log_probs
            else torch.zeros(batch_size, device=device),
            "actions": torch.stack(actions, dim=1) if actions else torch.zeros(batch_size[0], 0, device=device),
        }

        return out


class L2DPolicy4PPO(L2DPolicy):
    """
    L2DPolicy variant for PPO.
    For now, it's identical to L2DPolicy as the base class supports step-wise decoding.
    """

    pass
