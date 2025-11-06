import math
import torch
import torch.nn as nn

from .subnets import PointerEncoder, PointerDecoder


# Attention, Learn to Solve Routing Problems
class PointerNetwork(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=None,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization=None,
                 **kwargs):
        super(PointerNetwork, self).__init__()
        self.problem = problem
        assert problem.NAME == "tsp", "Pointer Network only supported for TSP"
        self.input_dim = 2
        self.encoder = PointerEncoder(embedding_dim, hidden_dim)
        self.decoder = PointerDecoder(
            embedding_dim,
            hidden_dim,
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits
        )

        # Trainable initial hidden states
        std = 1. / math.sqrt(embedding_dim)
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.decoder_in_0.data.uniform_(-std, std)

        self.embedding = nn.Parameter(torch.FloatTensor(self.input_dim, embedding_dim))
        self.embedding.data.uniform_(-std, std)

    def set_decode_type(self, decode_type):
        self.decoder.decode_type = decode_type

    def forward(self, inputs, eval_tours=None, return_pi=False):
        batch_size, graph_size, input_dim = inputs.size()
        embedded_inputs = torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            self.embedding
        ).view(graph_size, batch_size, -1)

        # query the actor net for the input indices 
        # making up the output, and the pointer attn 
        _log_p, pi = self._inner(embedded_inputs, eval_tours)

        cost, mask = self.problem.get_costs(inputs, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, inputs, eval_tours=None):
        encoder_hx = encoder_cx = torch.autograd.Variable(
            torch.zeros(1, inputs.size(1), self.encoder.hidden_dim, out=inputs.data.new()),
            requires_grad=False
        )

        # Encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # Repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input, inputs, dec_init_state, enc_h, eval_tours)
        return pointer_probs, input_idxs