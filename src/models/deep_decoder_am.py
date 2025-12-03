import math
import torch
import typing
import torch.nn as nn
import torch.nn.functional as F

from . import AttentionModel
from .subnets import GraphAttentionDecoder


class DeepAttentionModelFixed(typing.NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    mha_key: torch.Tensor
    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return DeepAttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                mha_key=self.mha_key[key]
            )
        return self[key]


class DeepDecoderAttentionModel(AttentionModel):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 encoder_class,
                 n_encode_layers=2,
                 n_encode_sublayers=None,
                 n_decode_layers=2,
                 dropout_rate=0.1,
                 aggregation="sum",
                 aggregation_graph="avg",
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 mask_graph=False,
                 normalization='batch',
                 norm_learn_affine=True,
                 norm_track_stats=False,
                 norm_eps_alpha=1e-05,
                 norm_momentum_beta=0.1,
                 lrnorm_k=1.0,
                 gnorm_groups=3,
                 activation_function='gelu',
                 af_param=1.0,
                 af_threshold=6.0,
                 af_replacement_value=6.0,
                 af_num_params=3,
                 af_uniform_range=[0.125, 1/3],
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 temporal_horizon=0,
                 predictor_layers=None):
        super(DeepDecoderAttentionModel, self).__init__(
            embedding_dim,
            hidden_dim,
            problem,
            encoder_class,
            n_encode_layers,
            n_encode_sublayers,
            dropout_rate,
            aggregation,
            aggregation_graph,
            tanh_clipping,
            mask_inner,
            mask_logits,
            mask_graph,
            normalization,
            norm_learn_affine,
            norm_track_stats,
            norm_eps_alpha,
            norm_momentum_beta,
            lrnorm_k,
            gnorm_groups,
            activation_function,
            af_param,
            af_threshold,
            af_replacement_value,
            af_num_params,
            af_uniform_range,
            n_heads,
            checkpoint_encoder,
            shrink_size,
            temporal_horizon
        )
        self.n_decode_layers = n_decode_layers
        self.decoder = GraphAttentionDecoder(
            n_heads=self.n_heads,
            embed_dim=self.embedding_dim,
            n_layers=self.n_decode_layers,
            feed_forward_hidden=self.hidden_dim,
            normalization=self.normalization,
            epsilon_alpha=self.epsilon_alpha,
            learn_affine=self.learn_affine,
            track_stats=self.track_stats,
            momentum_beta=self.momentum_beta,
            locresp_k=self.locresp_k,
            n_groups=self.n_groups,
            activation=self.activation,
            af_param=self.af_param,
            threshold=self.threshold,
            replacement_value=self.replacement_value,
            n_params=self.n_params,
            uniform_range=self.uniform_range,
            dropout_rate=self.dropout_rate
        )
        self.project_node_embeddings = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        if self.aggregation_graph == "avg":
            graph_embed = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        else:  # Default: disable graph embedding
            graph_embed = embeddings.sum(1) * 0.0

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        
        # The projection of the node embeddings for the attention is calculated once up front
        mha_key = self.project_node_embeddings(embeddings[:, None, :, :])
        
        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(mha_key, num_steps)
        )
        return DeepAttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed, state, normalize=True):
        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys for the nodes
        mha_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        graph_mask = None
        if self.mask_graph:
            # Compute the graph mask, for masking next action based on graph structure 
            graph_mask = state.get_edges_mask()

        # Compute logits (unnormalized log_p)
        log_p = self._one_to_many_logits(query, mha_K, mask, graph_mask)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _one_to_many_logits(self, query, mha_K, mask, graph_mask):
        logits = self.decoder(query, mha_K, graph_mask)

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -math.inf
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        return logits
    
    def _get_attention_node_data(self, fixed, state):
        if self.is_vrp and self.allow_partial:
            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            mha_key_step = self.project_node_step(state.demands_with_depot[:, :, :, None].clone())

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.mha_key + self._make_heads(mha_key_step)
            )

        # TSP or VRP without split delivery
        return fixed.mha_key
