from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict


class L2DDecoder(nn.Module):
    """
    Decoder for L2D Policy.
    Scores each job based on its embedding and global context.
    """

    def __init__(self, embed_dim: int, temp: float = 1.0, tanh_clipping: float = 10.0):
        super().__init__()
        self.proj_decoder = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.temp = temp
        self.tanh_clipping = tanh_clipping

    def forward(
        self,
        td: TensorDict,
        embeddings: Tuple[torch.Tensor, torch.Tensor],
        env: Optional[Any] = None,
        decode_type: str = "sampling",
        return_pi: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step decoding.

        Args:
            td: TensorDict containing current state
            embeddings: Tuple of (job_embeddings, full_embeddings) from encoder
            env: Environment (not used in simple projection decoder but good for interface)
            decode_type: "sampling" or "greedy"
            return_pi: If True, returns log_p of all actions (for PPO)

        Returns:
            (log_p, action) if return_pi is False
            (log_probs, action) if return_pi is True
        """
        job_embeddings, full_embeddings = embeddings

        # Global context (mean of all ops) -> (B, D)
        global_context = full_embeddings.mean(dim=1)

        # Expand context -> (B, J, D)
        ctx_expanded = global_context.unsqueeze(1).expand(-1, job_embeddings.size(1), -1)

        # Combined -> (B, J, 2D)
        combined = torch.cat([job_embeddings, ctx_expanded], dim=-1)

        # Logits -> (B, J)
        logits = self.proj_decoder(combined).squeeze(-1)

        # Masking
        mask = td["action_mask"]
        logits.masked_fill_(~mask, float("-inf"))

        # Clipping
        logits = self.tanh_clipping * torch.tanh(logits)

        # Probabilities
        log_p = torch.log_softmax(logits / self.temp, dim=-1)
        probs = log_p.exp()

        # Select action
        if decode_type == "greedy":
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)

        if return_pi:
            return log_p, action
        else:
            selected_log_prob = log_p.gather(1, action.unsqueeze(-1)).squeeze(-1)
            return selected_log_prob, action
