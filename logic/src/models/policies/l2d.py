from typing import Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.models.policies.common.constructive import ConstructivePolicy
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

        # Simple decoder: Score each job based on its embedding and global context
        self.proj_decoder = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

        self.temp = temp
        self.tanh_clipping = tanh_clipping

    def forward(
        self,
        td: TensorDict,
        env: Optional[Any] = None,
        decode_type: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> dict:
        """
        Forward pass.
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
            # job_embeddings: (B, J, D)
            # full_embeddings: (B, J*M, D) - useful for global context
            job_embeddings, full_embeddings = self.encoder(td)

            # Global context (mean of all ops or just next ops)
            # Use mean of all ops
            global_context = full_embeddings.mean(dim=1)  # (B, D)

            # Expand context
            # (B, 1, D) -> (B, J, D)
            ctx_expanded = global_context.unsqueeze(1).expand(-1, job_embeddings.size(1), -1)

            # Concatenate
            combined = torch.cat([job_embeddings, ctx_expanded], dim=-1)  # (B, J, 2D)

            # Compute logits
            logits = self.proj_decoder(combined).squeeze(-1)  # (B, J)

            # Masking
            mask = td["action_mask"]  # (B, J)
            logits.masked_fill_(~mask, float("-inf"))

            # Clipping
            logits = self.tanh_clipping * torch.tanh(logits)

            # Probabilities
            log_p = torch.log_softmax(logits / self.temp, dim=-1)
            probs = log_p.exp()

            # Select action
            if decode_type == "greedy":
                action = probs.argmax(dim=-1)
            elif decode_type == "sampling":
                action = torch.multinomial(probs, 1).squeeze(-1)
            else:
                action = probs.argmax(dim=-1)

            # Log prob of selected
            selected_log_prob = log_p.gather(1, action.unsqueeze(-1)).squeeze(-1)

            # Step
            td["action"] = action
            td = env.step(td)["next"]

            # Store
            actions.append(action)
            log_probs.append(selected_log_prob)

        # Stack
        out = {
            "reward": td["reward"].squeeze(-1),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1)
            if log_probs
            else torch.zeros(batch_size, device=device),
            "actions": torch.stack(actions, dim=1) if actions else torch.zeros(batch_size[0], 0, device=device),
        }

        return out
