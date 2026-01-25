"""Standard Attention Decoder for constructive routing problems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from logic.src.utils.functions.function import compute_in_batches


@dataclass
class AttentionModelFixed:
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key: Union[int, slice, torch.Tensor]) -> AttentionModelFixed:
        """Allows slicing the fixed context."""
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key],
            )
        # Fallback for integer indexing if needed, though usually used with tensors/slices
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key].unsqueeze(0),
            context_node_projected=self.context_node_projected[key].unsqueeze(0),
            glimpse_key=self.glimpse_key[:, key].unsqueeze(1),
            glimpse_val=self.glimpse_val[:, key].unsqueeze(1),
            logit_key=self.logit_key[key].unsqueeze(0),
        )


class AttentionDecoder(nn.Module):
    """
    Decoder based on Multi-Head Attention (MHA).

    Decodes the problem embedding into a sequence of nodes (visits) using
    an attention mechanism. Supports greedy decoding, sampling, and beam search.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        decode_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of input embeddings.
            hidden_dim: Dimension of hidden layers.
            problem: The problem instance (defines environment and constraints).
            n_heads: Number of attention heads.
            tanh_clipping: Clipping value for tanh in logits.
            mask_inner: Whether to mask invalid moves in the attention mechanism.
            mask_logits: Whether to mask invalid moves in the final logits.
            mask_graph: Whether to use graph masking.
            shrink_size: Threshold for shrinking the batch size (for completed instances).
            pomo_size: Number of starting nodes for POMO.
            spatial_bias: Whether to use spatial bias in attention.
            spatial_bias_scale: Scale factor for spatial bias.
            decode_type: Decoding strategy ('greedy', 'sampling').
        """
        super(AttentionDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.problem = problem
        self.is_wc = problem.NAME in ["wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"]
        self.is_vrpp = problem.NAME == "vrpp" or problem.NAME == "cvrpp"
        self.allow_partial = problem.NAME == "sdwcvrp"

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.mask_graph = mask_graph
        self.shrink_size = shrink_size
        self.pomo_size = pomo_size
        self.spatial_bias = spatial_bias
        self.spatial_bias_scale = spatial_bias_scale

        self.decode_type = decode_type

        # These layers were in AttentionModel
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(embedding_dim + (2 if self.is_wc else 1), embedding_dim, bias=False)

        if self.allow_partial:
            self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)

        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.temp: float = 1.0

    def set_step_context_dim(self, dim: int) -> None:
        """Sets the dimension of the step context projection."""
        self.project_step_context = nn.Linear(dim, self.embedding_dim, bias=False)

    def set_decode_type(self, decode_type: str, temp: Optional[float] = None) -> None:
        """Sets the decoding type and temperature."""
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(
        self,
        input: Union[torch.Tensor, dict[str, torch.Tensor]],
        embeddings: torch.Tensor,
        cost_weights: Optional[torch.Tensor] = None,
        dist_matrix: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper around _inner to match standard Module behavior.
        """
        # _inner returns (log_p, pi)
        return self._inner(
            input,
            None,
            embeddings,
            cost_weights,
            dist_matrix,
            profit_vars=kwargs.get("profit_vars"),
            mask=mask,
            expert_pi=expert_pi,
        )

    def _inner(
        self,
        nodes: Union[torch.Tensor, dict[str, torch.Tensor]],
        edges: Optional[torch.Tensor],
        embeddings: torch.Tensor,
        cost_weights: Optional[torch.Tensor],
        dist_matrix: Optional[torch.Tensor],
        profit_vars: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        sequences = []
        state = self.problem.make_state(
            nodes,
            edges,
            cost_weights,
            dist_matrix,
            profit_vars=profit_vars,
            hrl_mask=mask,
        )

        fixed = self._precompute(embeddings)

        i: int = 0
        batch_size = state.ids.size(0)
        while not (
            self.shrink_size is None and (state.all_finished() if expert_pi is None else i >= expert_pi.size(1))
        ):
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            mask_val = -math.inf if expert_pi is None else -50.0  # Use soft masking for expert evaluations
            log_p, step_mask = self._get_log_p(fixed, state, mask_val=mask_val)

            if mask is not None:
                if mask.size(1) == step_mask.size(2) - 1:
                    depot_mask = torch.zeros((mask.size(0), 1), dtype=torch.bool, device=mask.device)
                    mask_padded = torch.cat((depot_mask, mask), dim=1)
                else:
                    mask_padded = mask

                if step_mask.dim() == 3:
                    step_mask = step_mask | mask_padded.unsqueeze(1)
                else:
                    step_mask = step_mask | mask_padded

                if step_mask.dim() == 2:
                    current_mask = step_mask.unsqueeze(1)
                else:
                    current_mask = step_mask

                log_p = log_p.masked_fill(current_mask, mask_val)

            if expert_pi is not None:
                selected = expert_pi[:, i]
            else:
                selected = self._select_node(log_p.exp()[:, 0, :], step_mask[:, 0, :])

                if self.pomo_size > 0 and i == 0:
                    current_batch_size = selected.size(0)
                    B_val = current_batch_size // self.pomo_size
                    forced_selected = torch.arange(1, self.pomo_size + 1, device=selected.device).repeat(B_val)
                    selected = forced_selected

            state = state.update(selected)

            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            i += 1

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _select_node(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(
                1, selected.unsqueeze(-1)
            ).data.any(), "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings: torch.Tensor, num_steps: int = 1) -> AttentionModelFixed:
        graph_embed = embeddings.mean(1)  # Defaulting to avg for now, or use configured

        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous(),
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _make_heads(self, v: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(
                v.size(0),
                v.size(1) if num_steps is None else num_steps,
                v.size(2),
                self.n_heads,
                -1,
            )
            .permute(3, 0, 1, 2, 4)
        )

    def _get_log_p(
        self,
        fixed: AttentionModelFixed,
        state: Any,
        normalize: bool = True,
        mask_val: float = -math.inf,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = fixed.context_node_projected + self.project_step_context(
            self._get_parallel_step_context(fixed.node_embeddings, state)
        )

        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        mask = state.get_mask()

        graph_mask = None
        if self.mask_graph:
            graph_mask = state.get_edges_mask()

        dist_bias: Optional[torch.Tensor] = None
        if self.spatial_bias and state.dist_matrix is not None:
            dist_matrix = state.dist_matrix
            current_node = state.get_current_node()
            if current_node.dim() == 1:
                current_node = current_node.unsqueeze(-1)
            index = current_node.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))
            dist_bias = -self.spatial_bias_scale * dist_matrix.gather(1, index).detach()

        log_p, glimpse = self._one_to_many_logits(
            query,
            glimpse_K,
            glimpse_V,
            logit_K,
            mask,
            graph_mask,
            dist_bias=dist_bias,
            mask_val=mask_val,
        )
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _get_parallel_step_context(
        self, embeddings: torch.Tensor, state: Any, from_depot: bool = False
    ) -> torch.Tensor:
        current_node = state.get_current_node()
        if current_node.dim() == 1:
            current_node = current_node.unsqueeze(-1)
        batch_size, num_steps = current_node.size()

        if self.is_vrpp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1)),
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    state.get_current_profit()[:, :, None],
                ),
                -1,
            )
        elif self.is_wc:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1)),
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    state.get_current_efficiency()[:, :, None],
                    state.get_remaining_overflows()[:, :, None],
                ),
                -1,
            )
        else:
            assert False, "Unsupported problem"

    def _get_attention_node_data(
        self, fixed: AttentionModelFixed, state: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_wc and self.allow_partial:
            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = self.project_node_step(
                state.demands_with_depot[:, None, :, None].clone()
            ).chunk(3, dim=-1)
            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _one_to_many_logits(
        self,
        query: torch.Tensor,
        glimpse_K: torch.Tensor,
        glimpse_V: torch.Tensor,
        logit_K: torch.Tensor,
        mask: torch.Tensor,
        graph_mask: Optional[torch.Tensor] = None,
        dist_bias: Optional[torch.Tensor] = None,
        mask_val: float = -math.inf,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_steps, embed_dim = query.size()
        key_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if dist_bias is not None:
            compatibility += dist_bias.unsqueeze(0).unsqueeze(2)
        if self.mask_inner:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
            if self.mask_graph and graph_mask is not None:
                compatibility[graph_mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        val_size = embed_dim // self.n_heads
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if dist_bias is not None:
            logits += dist_bias.unsqueeze(1) if logits.size(1) > 1 else dist_bias

        if self.mask_logits and self.mask_graph and graph_mask is not None:
            logits[graph_mask] = mask_val
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = mask_val
        return logits, glimpse.squeeze(-2)

    def _calc_log_likelihood(
        self,
        _log_p: torch.Tensor,
        a: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_entropy: bool = False,
        kl_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if kl_loss:
            epsilon = 0.01
            n_actions = _log_p.size(-1)
            target_probs = torch.full_like(_log_p, epsilon / (n_actions - 1))
            target_probs.scatter_(-1, a.unsqueeze(-1), 1 - epsilon)
            log_target = target_probs.log()

            valid_mask = _log_p > -1e10
            # Use very small log_p for invalid masks to avoid exp(0)=1
            log_p_safe = torch.where(valid_mask, _log_p, torch.tensor(-100.0, device=_log_p.device))
            probs_safe = torch.where(valid_mask, _log_p.exp(), torch.zeros_like(_log_p))

            term = probs_safe * (log_p_safe - log_target)
            kl_div_safe = torch.where(valid_mask, term, torch.zeros_like(term))
            kl_div = kl_div_safe.sum(dim=-1)
            ll = -kl_div.sum(1)

            if return_entropy:
                entropy = -(log_p_safe * probs_safe).sum(dim=-1).sum(1)
                return ll, entropy
            return ll

        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Soft clamping for imitation.
        # If masked actions are selected (by an expert), we still want a gradient.
        # Standard masked _log_p has -inf. clamp(-inf, min) has 0 gradient.
        # We now handle this by allowing providing a 'soft_masking' flag or
        # just using a robust -inf replacement here if we detect it.

        if mask is not None:
            log_p[mask] = 0

        # If any log_p is -inf, it's because it was masked.
        # To get a gradient, we replace -inf with a large finite value
        # BEFORE the sum, but we do it in a way that preserves the gradient
        # of the underlying logits if possible...
        # Wait, the best way is to not mask in the first place or use label smoothing.
        # For now, we use a robust clamp that handles -inf by mapping it to a finite value.
        # Note: torch.clamp(-inf) is -50.
        log_p = torch.clamp(log_p, min=-50.0)

        ll = log_p.sum(1)
        if return_entropy:
            probs = _log_p.exp()
            log_p_safe = _log_p.clamp(min=-20)
            entropy = -(probs * log_p_safe).sum(-1).mean(1)
            return ll, entropy
        return ll

    def _get_log_p_topk(
        self,
        fixed: AttentionModelFixed,
        state: Any,
        k: Optional[int] = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets top-k log probabilities and their indices."""
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :],
        )

    def propose_expansions(
        self,
        beam: Any,
        fixed: AttentionModelFixed,
        expand_size: Optional[int] = None,
        normalize: bool = False,
        max_calc_batch_size: int = 4096,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Proposes expansions for beam search.

        Args:
            beam: The beam object.
            fixed: Fixed context.
            expand_size: Size to expand to.
            normalize: Whether to normalize probabilities.
            max_calc_batch_size: Max batch size for calculation.
        """
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size,
            beam,
            n=beam.size(),
        )
        assert log_p_topk.size(1) == 1, "Can only have single step"
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)
        feas_ind_2d = torch.nonzero(flat_feas)
        if len(feas_ind_2d) == 0:
            return None, None, None
        feas_ind = feas_ind_2d[:, 0]
        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]
