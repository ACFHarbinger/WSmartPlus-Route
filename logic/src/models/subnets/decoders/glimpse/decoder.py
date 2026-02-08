"""
Decoder based on Multi-Head Attention (MHA).

Decodes the problem embedding into a sequence of nodes (visits) using a
glimpse attention mechanism. Supports greedy decoding, sampling, and beam search.
"""

import math
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from .attention import make_heads, one_to_many_logits
from .fixed import AttentionModelFixed


class GlimpseDecoder(nn.Module):
    """
    Decoder based on Multi-Head Attention (MHA).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int = 8,
        mask_inner: bool = True,
        mask_logits: bool = True,
        tanh_clipping: float = 10.0,
        mask_graph: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        decode_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.problem = problem
        self.n_heads = n_heads
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        self.mask_graph = mask_graph
        self.shrink_size = shrink_size
        self.pomo_size = pomo_size
        self.spatial_bias = spatial_bias
        self.spatial_bias_scale = spatial_bias_scale
        self.decode_type = decode_type

        # Project node embeddings to MHA heads
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_step_context = nn.Linear(embed_dim, embed_dim, bias=False)

        # Step context dim can be overridden
        self.step_context_dim = embed_dim

    def set_step_context_dim(self, dim: int):
        """Sets the dimension of the step context projection."""
        self.step_context_dim = dim
        self.project_step_context = nn.Linear(dim, self.embed_dim, bias=False)

    def set_decode_type(self, decode_type: str, temp: Optional[float] = None):
        """Sets the decoding type and temperature."""
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(
        self,
        input: Union[torch.Tensor, dict[str, torch.Tensor]],
        embeddings: torch.Tensor,
        fixed_context: Optional[torch.Tensor] = None,
        init_context: Optional[torch.Tensor] = None,
        env: Optional[Any] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Standard Module forward wrapper."""
        return self._inner(
            input,
            embeddings,
            fixed_context,
            init_context,
            env,
            expert_pi,
            **kwargs,
        )

    def _inner(
        self,
        nodes: Union[torch.Tensor, dict[str, torch.Tensor]],
        embeddings: torch.Tensor,
        fixed_context: Optional[torch.Tensor] = None,
        init_context: Optional[torch.Tensor] = None,
        env: Optional[Any] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Constructive decoding loop."""
        outputs = []
        sequences = []

        # cost_weights and dist_matrix can be passed via kwargs or init_context
        cost_weights = kwargs.get("cost_weights")
        dist_matrix = kwargs.get("dist_matrix")

        state = self.problem.make_state(nodes, None, cost_weights, dist_matrix, **kwargs)
        fixed = self._precompute(embeddings)

        # Allow overriding decode_type via kwargs (e.g. from AttentionModel.forward)
        decode_type = kwargs.get("decode_type", self.decode_type)

        # Mask can be passed via kwargs
        mask = kwargs.get("mask")

        # Try to get graph size for safety break
        try:
            if isinstance(nodes, torch.Tensor):
                graph_size = nodes.shape[1]
            elif hasattr(nodes, "get"):
                # Handle TensorDict or dict
                loc_tensor = nodes.get("locs", nodes.get("loc"))
                graph_size = loc_tensor.shape[1] if hasattr(loc_tensor, "shape") else 100
            else:
                graph_size = 100
        except Exception:
            graph_size = 100

        # Safety break for infinite loops (e.g. 10x graph size)
        max_steps = max(100, graph_size * 10)

        i = 0
        while not state.all_finished() and i < max_steps:
            log_p, mask = self._get_log_p(fixed, state, mask=mask)
            selected = self._select_node(log_p.exp(), mask, decode_type=decode_type)

            state = state.update(selected)

            outputs.append(log_p)
            sequences.append(selected)
            i += 1

        if i >= max_steps:
            print(f" [!] Warning: Decoding reached max_steps ({max_steps}). Possible infinite loop.")

        # Stack outputs
        _log_p = torch.stack(outputs, 1)
        pi = torch.stack(sequences, 1)

        # Calculate cost
        cost = None
        if hasattr(self.problem, "get_costs"):
            out_cost = self.problem.get_costs(nodes, pi, None)
            # Handle tuple return (cost, mask, etc.)
            if isinstance(out_cost, tuple):
                cost = out_cost[0]
            else:
                cost = out_cost

        return _log_p, pi, cost

    def _select_node(self, probs: torch.Tensor, mask: Optional[torch.Tensor], decode_type: str = "greedy"):
        """Selection logic."""
        assert (probs == probs).all(), "Probs contain NaN"

        if decode_type == "greedy":
            _, selected = probs.max(1)
            if mask is not None:
                # Handle potential 3D mask [B, 1, N] -> [B, N]
                curr_mask = mask
                if curr_mask.dim() == 3:
                    curr_mask = curr_mask.squeeze(1)
                assert not curr_mask.gather(1, selected.unsqueeze(-1)).any(), "Selected masked node"
        elif decode_type == "sampling":
            selected = torch.multinomial(probs, 1).squeeze(1)

            # Mask handling for sampling loop check
            curr_mask = mask
            if curr_mask is not None and curr_mask.dim() == 3:
                curr_mask = curr_mask.squeeze(1)

            if curr_mask is not None:
                while curr_mask.gather(1, selected.unsqueeze(-1)).any():
                    selected = torch.multinomial(probs, 1).squeeze(1)
        else:
            raise ValueError(f"Unknown decode type: {decode_type}")

        return selected

    def _precompute(self, embeddings: torch.Tensor, num_steps: int = 1) -> AttentionModelFixed:
        """Precompute K,V for the attention mechanism."""
        node_embeddings = embeddings
        context_node_projected = self.project_fixed_context(node_embeddings.mean(1))

        # Joint projection
        qkv = self.project_node_embeddings(node_embeddings)
        glimpse_K, glimpse_V, logit_K = qkv.chunk(3, dim=-1)

        fixed = AttentionModelFixed(
            node_embeddings,
            context_node_projected,
            make_heads(glimpse_K, self.n_heads),
            make_heads(glimpse_V, self.n_heads),
            make_heads(logit_K, self.n_heads),
        )
        return fixed

    def _get_log_p(
        self,
        fixed: AttentionModelFixed,
        state: Any,
        normalize: bool = True,
        mask_val: float = -math.inf,
        mask: Optional[torch.Tensor] = None,
    ):
        """Compute log probabilities for the current state."""
        query = self._get_parallel_step_context(fixed.node_embeddings, state)

        # Logits: [batch_size, 1, graph_size]
        # Current implementation assumes single step
        if mask is None:
            mask = state.get_mask()

        logits = one_to_many_logits(
            query.unsqueeze(1),
            fixed.glimpse_key,
            fixed.glimpse_val,
            fixed.logit_key,
            mask.unsqueeze(1) if mask is not None and mask.dim() == 2 else mask,
            self.n_heads,
            tanh_clipping=self.tanh_clipping,
            mask_val=mask_val,
        ).squeeze(1)

        if normalize:
            return torch.log_softmax(logits, dim=-1), mask
        return logits, mask

    def _get_parallel_step_context(self, embeddings: torch.Tensor, state: Any, from_depot: bool = False):
        """Extract step context from state and project."""
        current_node = state.get_current_node()
        batch_size = embeddings.size(0)

        # Ensure current_node is (batch, 1) or (batch)
        if current_node.dim() > 1:
            current_node = current_node.reshape(batch_size, -1)
            # If it became (batch, 1), good. If (batch, N), typically N=1 here.
            current_node = current_node[:, 0]  # Make it (batch,) for cleaner unsqueezing below

        # Simple context: [batch, embed]
        # AM typically uses [embeddings(current_node), embeddings(first_node), capacity_left]
        # Here we follow a simplified version matching the original code
        step_context = embeddings.gather(
            1, current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim)
        ).squeeze(1)

        return self.project_step_context(step_context)

    def _calc_log_likelihood(
        self,
        _log_p: torch.Tensor,
        a: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_entropy: bool = False,
        kl_loss: bool = False,
    ):
        """Utility for loss calculation."""
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0

        if return_entropy:
            entropy = -(_log_p.exp() * _log_p).sum(2)
            if mask is not None:
                entropy[mask] = 0
            return log_p, entropy

        return log_p

    def propose_expansions(
        self,
        beam: Any,
        fixed: AttentionModelFixed,
        expand_size: Optional[int] = None,
        normalize: bool = False,
        max_calc_batch_size: int = 4096,
    ):
        """Proposals for beam search."""
        # Implementation depends on external Beam structure
        # Simplified for now
        log_p, _ = self._get_log_p(fixed, beam.state, normalize=normalize)
        return log_p
