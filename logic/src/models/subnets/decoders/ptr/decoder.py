"""Pointer Network Decoder.

This module provides the PointerDecoder, which implements a recurrent pointer
network architecture for combinatorial optimization problems.

Attributes:
    PointerDecoder: Recurrent decoder that "points" to input elements sequentially.

Example:
    >>> from logic.src.models.subnets.decoders.ptr.decoder import PointerDecoder
    >>> decoder = PointerDecoder(embed_dim=128, hidden_dim=128, ...)
    >>> (ll, pi), hidden = decoder(input, embeddings, hidden, context)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from .pointer_attention import PointerAttention


class PointerDecoder(nn.Module):
    """Standard Pointer Network Decoder.

    Implements the classical pointer network with an LSTM cell and attention-based
    pointing mechanism, as described in Vinyals et al. (2015).

    Attributes:
        embed_dim (int): Dimensionality of input embeddings.
        hidden_dim (int): Dimensionality of recurrent hidden state.
        n_glimpses (int): Number of glimpse attention steps per decoding step.
        mask_glimpses (bool): Whether to mask visited nodes during glimpses.
        mask_logits (bool): Whether to mask visited nodes in final logits.
        use_tanh (bool): Whether to use tanh in final attention.
        tanh_exploration (float): Tanh scaling constant.
        strategy (str): Decoding strategy (e.g., 'greedy', 'sampling').
        lstm (nn.LSTMCell): Recurrent transition cell.
        pointer (PointerAttention): Final pointing attention module.
        glimpse (PointerAttention): Glimpse attention module.
        sm (nn.Softmax): Softmax layer for glimpses.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        tanh_exploration: float,
        use_tanh: bool,
        n_glimpses: int = 1,
        mask_glimpses: bool = True,
        mask_logits: bool = True,
    ) -> None:
        """Initializes the PointerDecoder.

        Args:
            embed_dim: Embedding dimension.
            hidden_dim: Hidden dimension.
            tanh_exploration: Tanh exploration constant.
            use_tanh: Whether to use tanh.
            n_glimpses: Number of glimpses.
            mask_glimpses: Whether to mask glimpses.
            mask_logits: Whether to mask logits.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.strategy: Optional[str] = None

        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.pointer = PointerAttention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = PointerAttention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)

    def update_mask(self, mask: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """Updates the selection mask based on chosen nodes.

        Args:
            mask: Current boolean mask of shape (batch, nodes).
            selected: Indices of the selected nodes.

        Returns:
            torch.Tensor: Updated mask with selected nodes marked True.
        """
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def recurrence(
        self,
        x: torch.Tensor,
        h_in: Tuple[torch.Tensor, torch.Tensor],
        prev_mask: torch.Tensor,
        prev_idxs: Optional[torch.Tensor],
        step: int,
        context: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs one step of recurrent decoding.

        Args:
            x: Input embedding for the current step.
            h_in: Previous hidden and cell states.
            prev_mask: Mask from the previous step.
            prev_idxs: Selected indices from the previous step.
            step: Current decoding step index.
            context: Encoder outputs for attention referencing.

        Returns:
            Tuple: New hidden states, log probabilities, raw probabilities, and updated mask.
        """
        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()
        if not self.mask_logits:
            # Mask probabilities to prevent resampling if logits weren't masked
            probs[logit_mask] = 0.0

        return h_out, log_p, probs, logit_mask

    def calc_logits(
        self,
        x: torch.Tensor,
        h_in: Tuple[torch.Tensor, torch.Tensor],
        logit_mask: torch.Tensor,
        context: torch.Tensor,
        mask_glimpses: Optional[bool] = None,
        mask_logits: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculates attention logits for the next selection step.

        Args:
            x: Input feature vector.
            h_in: Current LSTM hidden and cell states.
            logit_mask: Current node selection mask.
            context: Attention reference sequence.
            mask_glimpses: Visibility flag for internal glimpses.
            mask_logits: Visibility flag for final pointing step.

        Returns:
            Tuple: Raw attention logits and updated hidden states.
        """
        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)
        for _ in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # Mask glimpses before softmax for readout stability
            if mask_glimpses:
                logits[logit_mask] = -np.inf
            # Matrix multiply soft-attention weights with references
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)

        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes output probabilities sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf

        return logits, h_out

    def forward(
        self,
        decoder_input: torch.Tensor,
        embedded_inputs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        eval_tours: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Autoregressively decodes a sequence of indices.

        Args:
            decoder_input: The initial starting input [batch_size, embed_dim].
            embedded_inputs: Reference embeddings [sourceL, batch, embed_dim].
            hidden: Initial hidden states [batch, hidden_dim].
            context: Encoder referencing pool [sourceL, batch, hidden_dim].
            eval_tours: Pre-defined sequence to evaluate (optional).

        Returns:
            Tuple: Sequence log-probabilities, chosen indices, and final hidden state.
        """
        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = embedded_inputs.data.new().bool().new(embedded_inputs.size(1), embedded_inputs.size(0)).zero_()

        for i in steps:
            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            # Select the next indices
            idxs = self.decode(probs, mask) if eval_tours is None else eval_tours[:, i]

            # Detach to prevent gradient issues during Reinforce estimation
            idxs = idxs.detach()

            # Gather input embedding of selected nodes
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:]),
            ).squeeze(0)

            outputs.append(log_p)
            selections.append(idxs)

        return (torch.stack(outputs, 1), torch.stack(selections, 1)), hidden

    def decode(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decodes probabilities to specific node indices based on strategy.

        Args:
            probs: Predicted probability distribution.
            mask: Current boolean mask of visited nodes.

        Returns:
            torch.Tensor: Selected node indices.

        Raises:
            AssertionError: If an unknown strategy is used or greedy action is invalid.
        """
        if self.strategy == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), (
                "Decode greedy: infeasible action has maximum probability"
            )
        elif self.strategy == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Handle potential race conditions on GPU resulting in invalid samples
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(" [!] resampling due to race condition")
                idxs = probs.multinomial(1).squeeze(1)
        else:
            raise AssertionError(f"Unknown strategy: {self.strategy}")

        return idxs
