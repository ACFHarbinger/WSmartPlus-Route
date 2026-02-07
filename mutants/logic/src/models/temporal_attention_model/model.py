"""
Temporal Attention Model for multi-day waste collection routing.

Extends the base AttentionModel with a GRU-based fill level predictor that uses
historical waste data to anticipate future bin fill levels. This enables proactive
collection decisions in stochastic demand scenarios (SDWCVRP, CWCVRP).

Architecture:
    Base AttentionModel + GatedRecurrentFillPredictor -> TemporalEmbedding -> CombineLayer

The temporal features are fused with static node embeddings before encoding,
allowing the attention mechanism to consider predicted future fill levels
when constructing routes.
"""

from typing import Any, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from logic.src.models.attention_model import AttentionModel
from logic.src.models.attention_model.decoding import DecodingMixin
from logic.src.models.attention_model.forward import ForwardMixin
from logic.src.models.attention_model.setup import SetupMixin
from logic.src.models.subnets.factories import NeuralComponentFactory


class TemporalAttentionModel(AttentionModel):
    """
    Attention Model extended with temporal features (waste history).

    Integrates a fill level predictor and processes historical waste data.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        component_factory: NeuralComponentFactory,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "mean",
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
        temporal_horizon: int = 5,
        predictor_layers: int = 2,
        pomo_size: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        connection_type: str = "residual",
        hyper_expansion: int = 4,
        decoder_type: str = "attention",
    ) -> None:
        """
        Initialize the Temporal Attention Model.
        """
        nn.Module.__init__(self)
        SetupMixin.__init__(self)
        ForwardMixin.__init__(self)
        DecodingMixin.__init__(self)
        # Use common init helpers, but note that for TemporalAM we set temporal_horizon=0
        # for the context embedder initially because we handle temporal features separately
        # in _get_initial_embeddings.
        self._init_parameters(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            n_heads=n_heads,
            pomo_size=pomo_size,
            checkpoint_encoder=checkpoint_encoder,
            aggregation_graph=aggregation_graph,
            temporal_horizon=0,
            tanh_clipping=tanh_clipping,
        )

        self._init_context_embedder(temporal_horizon=0)
        step_context_dim = self.context_embedder.step_context_dim

        self._init_components(
            component_factory=component_factory,
            step_context_dim=step_context_dim,
            n_encode_layers=n_encode_layers,
            n_encode_sublayers=n_encode_sublayers,
            n_decode_layers=n_decode_layers,
            dropout_rate=dropout_rate,
            predictor_layers=predictor_layers,
            normalization=normalization,
            norm_eps_alpha=norm_eps_alpha,
            norm_learn_affine=norm_learn_affine,
            norm_track_stats=norm_track_stats,
            norm_momentum_beta=norm_momentum_beta,
            lrnorm_k=lrnorm_k,
            gnorm_groups=gnorm_groups,
            activation_function=activation_function,
            af_param=af_param,
            af_threshold=af_threshold,
            af_replacement_value=af_replacement_value,
            af_num_params=af_num_params,
            af_uniform_range=af_uniform_range,
            aggregation=aggregation,
            connection_type=connection_type,
            hyper_expansion=hyper_expansion,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            mask_graph=mask_graph,
            shrink_size=shrink_size,
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
            decoder_type=decoder_type,
        )
        self.temporal_horizon = temporal_horizon
        from logic.src.models.subnets.modules import ActivationFunction
        from logic.src.models.subnets.other.grf_predictor import GatedRecurrentFillPredictor

        self.fill_predictor = GatedRecurrentFillPredictor(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout_rate,
        )

        self.temporal_embed = nn.Linear(1, embed_dim)
        if self.is_wc or self.is_vrpp:
            self.predict_future = True
        else:
            self.predict_future = False

        self.combine_embeddings = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            ActivationFunction(
                activation_function,
                af_param,
                af_threshold,
                af_replacement_value,
                af_num_params,
                cast(Tuple[float, float], tuple(af_uniform_range)) if af_uniform_range else None,
            ),
            nn.Linear(embed_dim, embed_dim),
        )

    def _get_initial_embeddings(self, input):
        """
        Get initial embeddings for nodes, incorporating predicted future fill levels.

        Args:
            input (dict): Input data containing node features and history.

        Returns:
            torch.Tensor: Combined embeddings (static + temporal).
        """
        # Get the base embeddings from context embedder (without temporal features)
        base_embeddings = self.context_embedder.init_node_embeddings(input)

        if "fill_history" not in list(input.keys()) or not self.predict_future:
            return base_embeddings

        fill_history = input["fill_history"]
        batch_size, graph_size, _ = fill_history.size()
        fill_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)

        predicted_fills = self.fill_predictor(fill_history)
        predicted_fills = predicted_fills.view(batch_size, graph_size, 1)

        # For depot node, set predicted fill to 0
        if self.is_vrpp or self.is_wc:
            depot_fill = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            predicted_fills = torch.cat((depot_fill, predicted_fills), dim=1)

        fill_embeddings = self.temporal_embed(predicted_fills)
        combined_embeddings = self.combine_embeddings(torch.cat((base_embeddings, fill_embeddings), dim=-1))
        return combined_embeddings, None

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
        Forward pass of the Temporal Attention Model.

        Handles temporal feature prediction and updates if needed.

        Args:
            input (dict): The input data dictionary.
            cost_weights (torch.Tensor, optional): Weights for different cost components. Defaults to None.
            return_pi (bool, optional): Whether to return the action sequence. Defaults to False.
            pad (bool, optional): Whether to pad the solution sequence. Defaults to False.
            mask (torch.Tensor, optional): Mask for valid actions. Defaults to None.
            expert_pi (torch.Tensor, optional): Expert policy. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: (cost, log_likelihood, cost_dict, pi, entropy)
        """
        if "fill_history" not in list(input.keys()) and self.predict_future:
            locs_key = "locs" if "locs" in input else "loc"
            batch_size = input[locs_key].size(0)
            graph_size = input[locs_key].size(1)

            # For VRP-like problems, adjust for depot (excluded from graph size)
            if self.is_vrpp or self.is_wc:
                pass  # graph_size is already correct (num customers)

            fill_history = torch.zeros(
                (batch_size, graph_size, self.temporal_horizon),
                device=input[locs_key].device,
            )
            input["fill_history"] = fill_history
        return super().forward(input, cost_weights, return_pi, pad, mask, expert_pi, **kwargs)

    def update_fill_history(self, fill_history, new_fills):
        """
        Update the fill history with new fill levels (rolling window).

        Args:
            fill_history (torch.Tensor): Current fill history.
            new_fills (torch.Tensor): New fill levels to append.

        Returns:
            torch.Tensor: Updated fill history.
        """
        updated_history = fill_history.clone()
        updated_history[:, :, :-1] = fill_history[:, :, 1:]
        updated_history[:, :, -1] = new_fills
        return updated_history

    def compute_simulator_day(self, input, graph, run_tsp=False):
        """
        Compute one simulation day, updating fill history if present.

        Args:
            input (dict): Input data.
            graph (object): The graph object.
            run_tsp (bool, optional): Whether to run TSP locally. Defaults to False.

        Returns:
            dict: Simulation results.
        """
        if "fill_history" in list(input.keys()) and "current_fill" in list(input.keys()):
            input["fill_history"] = self.update_fill_history(input["fill_history"], input["current_fill"])
        return super().compute_simulator_day(input, graph, run_tsp)
