"""
Core Attention Model for constructive routing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.src.models.context_embedder import VRPPContextEmbedder, WCContextEmbedder
from logic.src.models.model_factory import NeuralComponentFactory
from logic.src.utils.functions.beam_search import CachedLookup
from logic.src.utils.functions.function import sample_many


class AttentionModel(nn.Module):
    """
    Attention Model for Vehicle Routing Problems.

    This model uses an Encoder-Decoder architecture with Multi-Head Attention to solve
    various VRP instances (VRPP, WCVRP, CWCVRP). It encodes the problem graph and
    constructively decodes the solution one step at a time.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        problem: Any,
        component_factory: NeuralComponentFactory,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "avg",
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        normalization: str = "batch",
        norm_learn_affine: bool = True,
        norm_track_stats: bool = False,
        norm_eps_alpha: float = 1e-05,
        norm_momentum_beta: float = 0.1,
        lrnorm_k: float = 1.0,
        gnorm_groups: int = 3,
        activation_function: str = "gelu",
        af_param: float = 1.0,
        af_threshold: float = 6.0,
        af_replacement_value: float = 6.0,
        af_num_params: int = 3,
        af_uniform_range: List[float] = [0.125, 1 / 3],
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        temporal_horizon: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        predictor_layers: Optional[int] = None,
        connection_type: str = "residual",
        hyper_expansion: int = 4,
    ) -> None:
        """
        Initialize the Attention Model.

        Args:
            embedding_dim: Dimension of the embedding vectors.
            hidden_dim: Dimension of the hidden layers.
            problem: The problem instance wrapper (e.g., CVRRP, WCVRP).
            component_factory: Factory to create sub-components.
            n_encode_layers: Number of encoder layers. Defaults to 2.
            n_encode_sublayers: Number of sub-layers in encoder. Defaults to None.
            n_decode_layers: Number of decoder layers. Defaults to None.
            dropout_rate: Dropout rate. Defaults to 0.1.
            aggregation: Aggregation method. Defaults to "sum".
            aggregation_graph: Graph aggregation method. Defaults to "avg".
            tanh_clipping: Tanh clipping value for logits. Defaults to 10.0.
            mask_inner: Whether to mask inner attention. Defaults to True.
            mask_logits: Whether to mask logits. Defaults to True.
            mask_graph: Whether to mask graph attention. Defaults to False.
            normalization: Normalization type. Defaults to 'batch'.
            norm_learn_affine: Learn affine parameters in norm. Defaults to True.
            norm_track_stats: Track running stats in norm. Defaults to False.
            norm_eps_alpha: Epsilon/Alpha for norm. Defaults to 1e-05.
            norm_momentum_beta: Momentum/Beta for norm. Defaults to 0.1.
            lrnorm_k: K parameter for Local Response Norm. Defaults to 1.0.
            gnorm_groups: Groups for Group Norm. Defaults to 3.
            activation_function: Activation function name. Defaults to 'gelu'.
            af_param: Parameter for activation function. Defaults to 1.0.
            af_threshold: Threshold for activation function. Defaults to 6.0.
            af_replacement_value: Replacement value for activation function. Defaults to 6.0.
            af_num_params: Number of parameters for activation function. Defaults to 3.
            af_uniform_range: Uniform range for activation params. Defaults to [0.125, 1/3].
            n_heads: Number of attention heads. Defaults to 8.
            checkpoint_encoder: Whether to checkpoint encoder during training. Defaults to False.
            shrink_size: Size to shrink the problem graph to. Defaults to None.
            pomo_size: Size for POMO (Policy Optimization with Multiple Optima). Defaults to 0.
            temporal_horizon: Horizon for temporal features. Defaults to 0.
            spatial_bias: Whether to use spatial bias in attention. Defaults to False.
            spatial_bias_scale: Scale for spatial bias. Defaults to 1.0.
            entropy_weight: Weight for entropy regularization. Defaults to 0.0.
            predictor_layers: Number of layers in predictor. Defaults to None.
            connection_type: Connection type (e.g., 'residual'). Defaults to 'residual'.
            hyper_expansion: Expansion factor for hypernetworks. Defaults to 4.
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

    def set_decode_type(self, decode_type: str, temp: Optional[float] = None) -> None:
        """
        Set the decoding strategy for the model.

        Args:
            decode_type: The decoding strategy ('greedy' or 'sampling').
            temp: Temperature for sampling. Defaults to None.
        """
        if hasattr(self.decoder, "set_decode_type"):
            self.decoder.set_decode_type(decode_type, temp)  # type: ignore

    def _get_initial_embeddings(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get initial node embeddings from the context embedder.

        Args:
            input: The input data dictionary.

        Returns:
            Initial node embeddings.
        """
        return self.context_embedder.init_node_embeddings(input)

    def forward(
        self,
        input: Dict[str, Any],
        cost_weights: Optional[torch.Tensor] = None,
        return_pi: bool = False,
        pad: bool = False,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Forward pass of the Attention Model.

        Args:
            input: The input data containing problem state.
            cost_weights: Weights for different cost components. Defaults to None.
            return_pi: Whether to return the action probabilities/sequence. Defaults to False.
            pad: Whether to pad the solution sequence. Defaults to False.
            mask: Mask for valid actions. Defaults to None.
            expert_pi: Expert policy for KL divergence. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: (cost, log_likelihood, cost_dict, pi, entropy)
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
                    **kwargs,
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
                    **kwargs,
                )
                cost, cost_dict, mask = self.problem.get_costs(input, pi, cost_weights, dist_matrix)
        else:
            log_p, pi = self.decoder(
                input,
                embeddings,
                cost_weights,
                None,
                mask=mask,
                expert_pi=expert_pi,
                **kwargs,
            )
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
                pad_dim = input["locs"].size(1) + 1
                pi = F.pad(pi, (0, (pad_dim) - pi.size(-1)), value=0)
            return cost, ll, cost_dict, pi, entropy
        return cost, ll, cost_dict, None, entropy

    def beam_search(self, *args: Any, **kwargs: Any) -> Any:
        """
        Perform beam search decoding.

        Args:
            *args: Variable length argument list passed to the problem's beam_search.
            **kwargs: Arbitrary keyword arguments passed to the problem's beam_search.

        Returns:
            The result of the beam search.
        """
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input: Dict[str, torch.Tensor], edges: Optional[torch.Tensor]) -> CachedLookup:
        """
        Precompute fixed embeddings for the input.

        Args:
            input: The input data.
            edges: Edge information for the graph.

        Returns:
            A cached lookup object containing precomputed decoder state.
        """
        embeddings: torch.Tensor = self.embedder(self.context_embedder.init_node_embeddings(input), edges)
        return CachedLookup(self.decoder._precompute(embeddings))  # type: ignore

    def propose_expansions(
        self,
        beam: Any,
        fixed: Any,
        expand_size: Optional[int] = None,
        normalize: bool = False,
        max_calc_batch_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propose expansions for beam search.

        Args:
            beam: The current beam state.
            fixed: The precomputed fixed embeddings.
            expand_size: The number of expansions to propose. Defaults to None.
            normalize: Whether to normalize the probabilities. Defaults to False.
            max_calc_batch_size: Maximum batch size for calculation. Defaults to 4096.

        Returns:
            tuple: (log_p, mask)
        """
        return self.decoder.propose_expansions(beam, fixed, expand_size, normalize, max_calc_batch_size)  # type: ignore

    def sample_many(
        self,
        input: Dict[str, Any],
        cost_weights: Optional[torch.Tensor] = None,
        batch_rep: int = 1,
        iter_rep: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample multiple solutions for the same input (e.g., for POMO or validation).

        Args:
            input: The input data.
            cost_weights: Weights for different cost components. Defaults to None.
            batch_rep: Batch replication factor. Defaults to 1.
            iter_rep: Iteration replication factor. Defaults to 1.

        Returns:
            tuple: (costs, pis)
        """
        edges: Optional[torch.Tensor] = input.pop("edges") if "edges" in list(input.keys()) else None
        return sample_many(
            lambda x: self.decoder._inner(*x[:3], cost_weights, x[3]),  # type: ignore
            lambda x, pi: self.problem.get_costs(x[0], pi, cost_weights)[:2],
            (
                input,
                edges,
                self.embedder(self.context_embedder.init_node_embeddings(input), edges),
                input.get("dist_matrix", None),
            ),
            batch_rep,
            iter_rep,
        )
