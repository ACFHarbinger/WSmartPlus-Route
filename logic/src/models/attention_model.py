"""
This module contains the Attention Model implementation for solving Vehicle Routing Problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.src.models.context_embedder import VRPPContextEmbedder, WCContextEmbedder
from logic.src.models.model_factory import NeuralComponentFactory
from logic.src.utils.beam_search import CachedLookup
from logic.src.utils.functions import sample_many


class AttentionModel(nn.Module):
    """
    Attention Model for Vehicle Routing Problems.

    This model uses an Encoder-Decoder architecture with Multi-Head Attention to solve
    various VRP instances (VRPP, WCVRP, CWCVRP). It encodes the problem graph and
    constructively decodes the solution one step at a time.
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        component_factory,
        n_encode_layers=2,
        n_encode_sublayers=None,
        n_decode_layers=None,
        dropout_rate=0.1,
        aggregation="sum",
        aggregation_graph="avg",
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        normalization="batch",
        norm_learn_affine=True,
        norm_track_stats=False,
        norm_eps_alpha=1e-05,
        norm_momentum_beta=0.1,
        lrnorm_k=1.0,
        gnorm_groups=3,
        activation_function="gelu",
        af_param=1.0,
        af_threshold=6.0,
        af_replacement_value=6.0,
        af_num_params=3,
        af_uniform_range=[0.125, 1 / 3],
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
        pomo_size=0,
        temporal_horizon=0,
        spatial_bias=False,
        spatial_bias_scale=1.0,
        entropy_weight=0.0,
        predictor_layers=None,
        connection_type="residual",
        hyper_expansion=4,
    ):
        """
        Initialize the Attention Model.

        Args:
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_dim (int): Dimension of the hidden layers.
            problem (object): The problem instance wrapper (e.g., CVRRP, WCVRP).
            component_factory (NeuralComponentFactory): Factory to create sub-components.
            n_encode_layers (int, optional): Number of encoder layers. Defaults to 2.
            n_encode_sublayers (int, optional): Number of sub-layers in encoder. Defaults to None.
            n_decode_layers (int, optional): Number of decoder layers. Defaults to None.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            aggregation (str, optional): Aggregation method. Defaults to "sum".
            aggregation_graph (str, optional): Graph aggregation method. Defaults to "avg".
            tanh_clipping (float, optional): Tanh clipping value for logits. Defaults to 10.0.
            mask_inner (bool, optional): Whether to mask inner attention. Defaults to True.
            mask_logits (bool, optional): Whether to mask logits. Defaults to True.
            mask_graph (bool, optional): Whether to mask graph attention. Defaults to False.
            normalization (str, optional): Normalization type. Defaults to 'batch'.
            norm_learn_affine (bool, optional): Learn affine parameters in norm. Defaults to True.
            norm_track_stats (bool, optional): Track running stats in norm. Defaults to False.
            norm_eps_alpha (float, optional): Epsilon/Alpha for norm. Defaults to 1e-05.
            norm_momentum_beta (float, optional): Momentum/Beta for norm. Defaults to 0.1.
            lrnorm_k (float, optional): K parameter for Local Response Norm. Defaults to 1.0.
            gnorm_groups (int, optional): Groups for Group Norm. Defaults to 3.
            activation_function (str, optional): Activation function name. Defaults to 'gelu'.
            af_param (float, optional): Parameter for activation function. Defaults to 1.0.
            af_threshold (float, optional): Threshold for activation function. Defaults to 6.0.
            af_replacement_value (float, optional): Replacement value for activation function. Defaults to 6.0.
            af_num_params (int, optional): Number of parameters for activation function. Defaults to 3.
            af_uniform_range (list, optional): Uniform range for activation params. Defaults to [0.125, 1/3].
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            checkpoint_encoder (bool, optional): Whether to checkpoint encoder during training. Defaults to False.
            shrink_size (int, optional): Size to shrink the problem graph to. Defaults to None.
            pomo_size (int, optional): Size for POMO (Policy Optimization with Multiple Optima). Defaults to 0.
            temporal_horizon (int, optional): Horizon for temporal features. Defaults to 0.
            spatial_bias (bool, optional): Whether to use spatial bias in attention. Defaults to False.
            spatial_bias_scale (float, optional): Scale for spatial bias. Defaults to 1.0.
            entropy_weight (float, optional): Weight for entropy regularization. Defaults to 0.0.
            predictor_layers (int, optional): Number of layers in predictor. Defaults to None.
            connection_type (str, optional): Connection type (e.g., 'residual'). Defaults to 'residual'.
            hyper_expansion (int, optional): Expansion factor for hypernetworks. Defaults to 4.
        """
        super(AttentionModel, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.problem = problem
        self.pomo_size = pomo_size
        self.checkpoint_encoder = checkpoint_encoder
        self.aggregation_graph = aggregation_graph
        self.temporal_horizon = temporal_horizon

        # Initialize Context Embedder Strategy
        self.is_wc = (
            problem.NAME == "wcvrp"
            or problem.NAME == "cwcvrp"
            or problem.NAME == "sdwcvrp"
            or problem.NAME == "scwcvrp"
        )
        self.is_vrpp = problem.NAME == "vrpp" or problem.NAME == "cvrpp"
        node_dim = 3
        if self.is_wc:
            self.context_embedder = WCContextEmbedder(
                embedding_dim, node_dim=node_dim, temporal_horizon=temporal_horizon
            )
        else:
            self.context_embedder = VRPPContextEmbedder(
                embedding_dim, node_dim=node_dim, temporal_horizon=temporal_horizon
            )

        step_context_dim = self.context_embedder.step_context_dim

        # Use Factory to create components
        if not isinstance(component_factory, NeuralComponentFactory):
            pass

        encoder_kwargs = {
            "n_heads": self.n_heads,
            "embed_dim": self.embedding_dim,
            "n_layers": n_encode_layers,
            "n_sublayers": n_encode_sublayers,
            "feed_forward_hidden": self.hidden_dim,
            "normalization": normalization,
            "epsilon_alpha": norm_eps_alpha,
            "learn_affine": norm_learn_affine,
            "track_stats": norm_track_stats,
            "momentum_beta": norm_momentum_beta,
            "locresp_k": lrnorm_k,
            "n_groups": gnorm_groups,
            "activation": activation_function,
            "af_param": af_param,
            "threshold": af_threshold,
            "replacement_value": af_replacement_value,
            "n_params": af_num_params,
            "uniform_range": af_uniform_range,
            "dropout_rate": dropout_rate,
            "agg": aggregation,
            "connection_type": connection_type,
            "expansion_rate": hyper_expansion,
        }

        self.embedder = component_factory.create_encoder(**encoder_kwargs)

        self.decoder = component_factory.create_decoder(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            problem=problem,
            n_heads=self.n_heads,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            mask_graph=mask_graph,
            shrink_size=shrink_size,
            pomo_size=pomo_size,
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
        )

        # Configure decoder step context
        if hasattr(self.decoder, "set_step_context_dim"):
            self.decoder.set_step_context_dim(step_context_dim)

    def set_decode_type(self, decode_type, temp=None):
        """
        Set the decoding strategy for the model.

        Args:
            decode_type (str): The decoding strategy ('greedy' or 'sampling').
            temp (float, optional): Temperature for sampling. Defaults to None.
        """
        self.decoder.set_decode_type(decode_type, temp)

    def _get_initial_embeddings(self, input):
        """
        Get initial node embeddings from the context embedder.

        Args:
            input (dict): The input data dictionary.

        Returns:
            torch.Tensor: Initial node embeddings.
        """
        return self.context_embedder.init_node_embeddings(input)

    def forward(
        self,
        input,
        cost_weights=None,
        return_pi=False,
        pad=False,
        mask=None,
        expert_pi=None,
        **kwargs,
    ):
        """
        Forward pass of the Attention Model.

        Args:
            input (dict): The input data containing problem state.
            cost_weights (torch.Tensor, optional): Weights for different cost components. Defaults to None.
            return_pi (bool, optional): Whether to return the action probabilities/sequence. Defaults to False.
            pad (bool, optional): Whether to pad the solution sequence. Defaults to False.
            mask (torch.Tensor, optional): Mask for valid actions. Defaults to None.
            expert_pi (torch.Tensor, optional): Expert policy for KL divergence. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: (cost, log_likelihood, cost_dict, pi, entropy)
                   - cost: The total cost of the solution (negative reward).
                   - log_likelihood: Log-likelihood of the solution.
                   - cost_dict: Dictionary of individual cost components.
                   - pi (optional): The sequence of selected nodes.
                   - entropy (optional): Entropy of the policy.
        """
        edges = input.get("edges", None)
        dist_matrix = input.get("dist", None)  # Using 'dist' key consistent with original

        node_embeddings = self._get_initial_embeddings(input)

        if self.checkpoint_encoder and self.training:
            embeddings = torch.utils.checkpoint.checkpoint(self.embedder, node_embeddings, edges, use_reentrant=False)
        else:
            if getattr(self.embedder, "init_edge_embed", None) is not None:
                embeddings = self.embedder(node_embeddings, edges, dist=dist_matrix)
            else:
                embeddings = self.embedder(node_embeddings, edges)

        if dist_matrix is not None:
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)

            if self.pomo_size > 0:

                def expand(t):
                    """
                    Expand tensor or dictionary of tensors for POMO.

                    Args:
                        t (torch.Tensor or dict or None): Input to expand.

                    Returns:
                        Expanded input.
                    """
                    if t is None:
                        return None
                    if isinstance(t, torch.Tensor):
                        return t.repeat_interleave(self.pomo_size, dim=0)
                    if isinstance(t, dict):
                        return {k: expand(v) for k, v in t.items()}
                    return t

                expanded_input = expand(input)
                expanded_embeddings = expand(embeddings)

                if dist_matrix.size(0) == 1:
                    expanded_dist_matrix = dist_matrix.expand(expanded_embeddings.size(0), -1, -1)
                else:
                    expanded_dist_matrix = dist_matrix.repeat_interleave(self.pomo_size, dim=0)

                expanded_mask = expand(mask)

                log_p, pi = self.decoder(
                    expanded_input,
                    expanded_embeddings,
                    cost_weights,
                    expanded_dist_matrix,
                    mask=expanded_mask,
                    expert_pi=expert_pi,
                )
                cost, cost_dict, mask = self.problem.get_costs(expanded_input, pi, cost_weights, expanded_dist_matrix)
            else:
                if dist_matrix.size(0) == 1 and embeddings.size(0) > 1:
                    dist_matrix = dist_matrix.expand(embeddings.size(0), -1, -1)
                log_p, pi = self.decoder(
                    input,
                    embeddings,
                    cost_weights,
                    dist_matrix,
                    mask=mask,
                    expert_pi=expert_pi,
                )
                cost, cost_dict, mask = self.problem.get_costs(input, pi, cost_weights, dist_matrix)
        else:
            log_p, pi = self.decoder(input, embeddings, cost_weights, None, mask=mask, expert_pi=expert_pi)
            cost, cost_dict, mask = self.problem.get_costs(input, pi, cost_weights, None)

        use_kl = kwargs.get("kl_loss", False) and expert_pi is not None
        if expert_pi is not None and use_kl:
            res = self.decoder._calc_log_likelihood(log_p, expert_pi, mask, return_entropy=self.training, kl_loss=True)
        else:
            res = self.decoder._calc_log_likelihood(log_p, pi, mask, return_entropy=self.training, kl_loss=False)

        if self.training:
            ll, entropy = res
        else:
            ll = res
            entropy = None

        if return_pi:
            if pad:
                pad_dim = input["loc"].size(1) + 1
                pi = F.pad(pi, (0, (pad_dim) - pi.size(-1)), value=0)
            return cost, ll, cost_dict, pi, entropy
        return cost, ll, cost_dict, None, entropy

    def beam_search(self, *args, **kwargs):
        """
        Perform beam search decoding.

        Args:
            *args: Variable length argument list passed to the problem's beam_search.
            **kwargs: Arbitrary keyword arguments passed to the problem's beam_search.

        Returns:
            list: The result of the beam search.
        """
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input, edges):
        """
        Precompute fixed embeddings for the input.

        Args:
            input (dict): The input data.
            edges (torch.Tensor): Edge information for the graph.

        Returns:
            CachedLookup: A cached lookup object containing precomputed decoder state.
        """
        embeddings = self.embedder(self.context_embedder.init_node_embeddings(input), edges)
        return CachedLookup(self.decoder._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        """
        Propose expansions for beam search.

        Args:
            beam (object): The current beam state.
            fixed (object): The precomputed fixed embeddings.
            expand_size (int, optional): The number of expansions to propose. Defaults to None.
            normalize (bool, optional): Whether to normalize the probabilities. Defaults to False.
            max_calc_batch_size (int, optional): Maximum batch size for calculation. Defaults to 4096.

        Returns:
            tuple: (log_p, mask)
                   - log_p: Log probabilities of the expansions.
                   - mask: Mask of valid expansions.
        """
        return self.decoder.propose_expansions(beam, fixed, expand_size, normalize, max_calc_batch_size)

    def sample_many(self, input, cost_weights=None, batch_rep=1, iter_rep=1):
        """
        Sample multiple solutions for the same input (e.g., for POMO or validation).

        Args:
            input (dict): The input data.
            cost_weights (torch.Tensor, optional): Weights for different cost components. Defaults to None.
            batch_rep (int, optional): Batch replication factor. Defaults to 1.
            iter_rep (int, optional): Iteration replication factor. Defaults to 1.

        Returns:
            tuple: (costs, pis)
                   - costs: Costs for all sampled solutions.
                   - pis: Sequences of actions for all solutions.
        """
        edges = input.pop("edges") if "edges" in input.keys() else None
        # This requires decoder delegation
        return sample_many(
            lambda input: self.decoder._inner(*input[:3], cost_weights, input[3]),
            lambda input, pi: self.problem.get_costs(input[0], pi, cost_weights)[:2],
            (
                input,
                edges,
                self.embedder(self.context_embedder.init_node_embeddings(input), edges),
                input.get("dist_matrix", None),
            ),
            batch_rep,
            iter_rep,
        )
