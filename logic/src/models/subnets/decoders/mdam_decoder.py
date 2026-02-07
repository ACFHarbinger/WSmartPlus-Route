"""
MDAM Decoder with Multi-Path Decoding and KL Divergence.

Multi-path decoder that constructs diverse solutions by running multiple
decoder paths in parallel and computing KL divergence for diversity loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.embeddings import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.embeddings.dynamic_embedding import DynamicEmbedding


def _decode_probs(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor],
    decode_type: str = "sampling",
) -> torch.Tensor:
    """
    Decode action from probability distribution.

    Args:
        probs: Probability distribution (batch, num_nodes).
        mask: Valid action mask (batch, num_nodes).
        decode_type: 'greedy' or 'sampling'.

    Returns:
        Selected actions (batch,).
    """
    if mask is not None:
        probs = probs.masked_fill(~mask, 0.0)
        # Renormalize
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    if decode_type == "greedy":
        return probs.argmax(dim=-1)
    else:
        # Sampling
        probs = probs.clamp(min=1e-8)
        return torch.multinomial(probs, 1).squeeze(-1)


@dataclass
class PrecomputedCache:
    """Cache for precomputed encoder outputs."""

    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


class MDAMDecoder(nn.Module):
    """
    MDAM Multi-Path Decoder.

    Constructs diverse solutions by running multiple decoder paths in parallel.
    Each path has its own context embedding and projection layers.
    KL divergence between path outputs encourages solution diversity.

    Reference:
        Xin et al. "Multi-Decoder Attention Model with Embedding Glimpse for
        Solving Vehicle Routing Problems" (AAAI 2021)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_paths: int = 5,
        env_name: str = "vrpp",
        mask_inner: bool = True,
        mask_logits: bool = True,
        eg_step_gap: int = 200,
        tanh_clipping: float = 10.0,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
    ) -> None:
        """
        Initialize MDAM decoder.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_paths: Number of parallel decoder paths for diversity.
            env_name: Environment name for context embedding selection.
            mask_inner: Whether to mask inner attention.
            mask_logits: Whether to mask final logits.
            eg_step_gap: Gap between encoder update steps during decoding.
            tanh_clipping: Clipping value for tanh on logits.
            train_decode_type: Decoding type during training.
            val_decode_type: Decoding type during validation.
            test_decode_type: Decoding type during testing.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_paths = num_paths
        self.env_name = env_name
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.eg_step_gap = eg_step_gap
        self.tanh_clipping = tanh_clipping

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        # Dynamic embedding for step context
        self.dynamic_embedding = DynamicEmbedding(embed_dim=embed_dim)

        # Placeholder for first step
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        # Path-specific modules
        self.context = nn.ModuleList([self._get_context_embedding(env_name, embed_dim) for _ in range(num_paths)])

        self.project_node_embeddings = nn.ModuleList(
            [nn.Linear(embed_dim, 3 * embed_dim, bias=False) for _ in range(num_paths)]
        )

        self.project_fixed_context = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_paths)]
        )

        self.project_step_context = nn.ModuleList(
            [nn.Linear(2 * embed_dim, embed_dim, bias=False) for _ in range(num_paths)]
        )

        self.project_out = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_paths)])

    def _get_context_embedding(self, env_name: str, embed_dim: int) -> nn.Module:
        """Get environment-specific context embedding."""
        if env_name in CONTEXT_EMBEDDING_REGISTRY:
            return CONTEXT_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim)
        # Fallback to simple context
        return nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        decode_type: str = "greedy",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MDAM Multi-path decoding.
        """
        # Unpack MDAM specific embeddings
        h, graph_embed, attn, V, h_old = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            embeddings,
        )
        encoder = kwargs.get("encoder")

        if decode_type is None:
            decode_type = self.train_decode_type

        # First step: calculate KL divergence between paths
        kl_divergence = self._compute_initial_kl_divergence(td, h)

        # Main decoding for each path
        reward_list = []
        ll_list = []
        action_list = []

        for path_idx in range(self.num_paths):
            reward, ll, actions = self._decode_path(
                td.clone(),
                h.clone(),
                env,
                attn.clone(),
                V.clone(),
                h_old.clone(),
                encoder,
                path_idx,
                decode_type,
            )
            reward_list.append(reward)
            ll_list.append(ll)
            action_list.append(actions)

        # Stack results
        reward = torch.stack(reward_list, dim=1)  # (batch, num_paths)
        log_likelihood = torch.stack(ll_list, dim=1)  # (batch, num_paths)

        # Return actions from first path (or could select best)
        actions = action_list[0]

        return reward, log_likelihood, kl_divergence, actions

    def _compute_initial_kl_divergence(
        self,
        td: TensorDict,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between path logprobs at first step."""
        if self.num_paths <= 1:
            return torch.tensor(0.0, device=h.device)

        output_list = []
        for path_idx in range(self.num_paths):
            fixed = self._precompute(h, path_index=path_idx)
            logprobs, _ = self._get_logprobs(fixed, td, path_idx)

            # Ensure numerical stability
            logprobs = torch.clamp(logprobs, min=-1e9)
            output_list.append(logprobs[:, 0, :])

        # Compute pairwise KL divergence
        kl_divergences = []
        for i in range(self.num_paths):
            for j in range(self.num_paths):
                if i == j:
                    continue
                # KL(P_i || P_j) = sum(P_i * (log P_i - log P_j))
                kl = torch.sum(
                    torch.exp(output_list[i]) * (output_list[i] - output_list[j]),
                    dim=-1,
                )
                kl_divergences.append(kl)

        return torch.stack(kl_divergences, dim=0).mean()

    def _decode_path(
        self,
        td: TensorDict,
        h: torch.Tensor,
        env: RL4COEnvBase,
        attn: torch.Tensor,
        V: torch.Tensor,
        h_old: torch.Tensor,
        encoder: Any,
        path_idx: int,
        decode_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a single path."""
        outputs = []
        actions = []

        fixed = self._precompute(h, path_index=path_idx)
        step = 0

        while not td["done"].all():
            # Periodic encoder update
            if step > 1 and step % self.eg_step_gap == 0:
                mask = td.get("action_mask", None)
                if mask is not None and hasattr(encoder, "change"):
                    # Update embeddings with current mask
                    h, _ = encoder.change(attn, V, h_old, mask)
                    fixed = self._precompute(h, path_index=path_idx)

            # Get logprobs and mask
            logprobs, mask = self._get_logprobs(fixed, td, path_idx)

            # Store first mask if needed for future VRP-specific logic
            _ = mask.clone() if step == 0 and mask is not None else None

            # Select action
            probs = torch.exp(logprobs[:, 0, :])
            action = _decode_probs(probs, mask, decode_type=decode_type)

            # Step environment
            td.set("action", action)
            td = env.step(td)["next"]

            outputs.append(logprobs[:, 0, :])
            actions.append(action)
            step += 1

        # Compute reward and log-likelihood
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, graph_size)
        actions = torch.stack(actions, dim=1)  # (batch, seq_len)

        reward = env.get_reward(td, actions)
        ll = self._get_log_likelihood(outputs, actions)

        return reward, ll, actions

    def _precompute(
        self,
        h_embed: torch.Tensor,
        num_steps: int = 1,
        path_index: int = 0,
    ) -> PrecomputedCache:
        """Precompute fixed projections."""
        graph_embed = h_embed.mean(dim=1)

        # Fixed context projection
        fixed_context = self.project_fixed_context[path_index](graph_embed)[:, None, :]

        # Node embedding projections
        projected = self.project_node_embeddings[path_index](h_embed[:, None, :, :])
        glimpse_key, glimpse_val, logit_key = projected.chunk(3, dim=-1)

        return PrecomputedCache(
            node_embeddings=h_embed,
            graph_context=fixed_context,
            glimpse_key=self._make_heads(glimpse_key, num_steps),
            glimpse_val=self._make_heads(glimpse_val, num_steps),
            logit_key=logit_key.contiguous(),
        )

    def _make_heads(
        self,
        v: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Reshape for multi-head attention."""
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.num_heads, -1)
            .expand(
                v.size(0),
                v.size(1) if num_steps is None else num_steps,
                v.size(2),
                self.num_heads,
                -1,
            )
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch, num_steps, graph_size, head_dim)
        )

    def _get_logprobs(
        self,
        fixed: PrecomputedCache,
        td: TensorDict,
        path_index: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get log probabilities for next action."""
        # Step context from current state
        context_mod = self.context[path_index]
        if hasattr(context_mod, "forward"):
            step_context = context_mod(fixed.node_embeddings, td)
        else:
            step_context = context_mod(fixed.node_embeddings.mean(dim=1))

        glimpse_q = fixed.graph_context + step_context.unsqueeze(1)

        # Dynamic embedding contribution
        glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn = self.dynamic_embedding(td)

        glimpse_k = fixed.glimpse_key + glimpse_key_dyn
        glimpse_v = fixed.glimpse_val + glimpse_val_dyn
        logit_k = fixed.logit_key + logit_key_dyn

        # Get mask
        mask = td.get("action_mask", None)

        # Compute logits
        logprobs, _ = self._compute_logits(glimpse_q, glimpse_k, glimpse_v, logit_k, mask, path_index)

        return logprobs, mask

    def _compute_logits(
        self,
        query: torch.Tensor,
        glimpse_K: torch.Tensor,
        glimpse_V: torch.Tensor,
        logit_K: torch.Tensor,
        mask: Optional[torch.Tensor],
        path_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-based logits."""
        batch_size, num_steps, embed_dim = query.size()
        key_size = embed_dim // self.num_heads

        # Reshape query for multi-head
        glimpse_Q = query.view(batch_size, num_steps, self.num_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Compute compatibility
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(key_size)

        if self.mask_inner and mask is not None:
            compatibility_mask = ~mask[None, :, None, None, :].expand_as(compatibility)
            compatibility[compatibility_mask] = -math.inf

        # Compute attention and aggregate
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project out
        glimpse = self.project_out[path_index](
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.num_heads * key_size)
        )

        # Compute final logits
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        # Apply tanh clipping
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        # Apply mask
        if self.mask_logits and mask is not None:
            logits[~mask[:, None, :]] = -math.inf

        # Convert to log probs
        logprobs = F.log_softmax(logits, dim=-1)

        return logprobs, glimpse.squeeze(-2)

    def _get_log_likelihood(
        self,
        log_probs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-likelihood of action sequence."""
        # log_probs: (batch, seq_len, num_nodes)
        # actions: (batch, seq_len)
        ll = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return ll.sum(dim=-1)
