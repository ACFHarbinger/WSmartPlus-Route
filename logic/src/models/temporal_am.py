"""
This module contains the Temporal Attention Model implementation.
"""

import torch
import torch.nn as nn

from . import AttentionModel


class TemporalAttentionModel(AttentionModel):
    """
    Attention Model extended with temporal features (waste history).

    Integrates a fill level predictor and processes historical waste data.
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
        aggregation_graph="mean",
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
        temporal_horizon=5,
        predictor_layers=2,
    ):
        """
        Initialize the Temporal Attention Model.

        Args:
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_dim (int): Dimension of the hidden layers.
            problem (object): The problem instance wrapper.
            component_factory (NeuralComponentFactory): Factory to create sub-components.
            n_encode_layers (int, optional): Number of encoder layers. Defaults to 2.
            n_encode_sublayers (int, optional): Number of sub-layers in encoder. Defaults to None.
            n_decode_layers (int, optional): Number of decoder layers. Defaults to None.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            aggregation (str, optional): Aggregation method. Defaults to "sum".
            aggregation_graph (str, optional): Graph aggregation method. Defaults to "mean".
            tanh_clipping (float, optional): Tanh clipping value. Defaults to 10.0.
            mask_inner (bool, optional): Whether to mask inner attention. Defaults to True.
            mask_logits (bool, optional): Whether to mask logits. Defaults to True.
            mask_graph (bool, optional): Whether to mask graph attention. Defaults to False.
            normalization (str, optional): Normalization type. Defaults to 'batch'.
            norm_learn_affine (bool, optional): Learn affine parameters. Defaults to True.
            norm_track_stats (bool, optional): Track running stats. Defaults to False.
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
            checkpoint_encoder (bool, optional): Whether to checkpoint encoder. Defaults to False.
            shrink_size (int, optional): Size to shrink the problem graph to. Defaults to None.
            temporal_horizon (int, optional): Horizon for temporal features. Defaults to 5.
            predictor_layers (int, optional): Number of layers in predictor. Defaults to 2.
        """
        super(TemporalAttentionModel, self).__init__(
            embedding_dim,
            hidden_dim,
            problem,
            component_factory,
            n_encode_layers,
            n_encode_sublayers,
            n_decode_layers,
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
            shrink_size=shrink_size,
            pomo_size=0,
            temporal_horizon=0,
        )
        self.temporal_horizon = temporal_horizon
        from . import GatedRecurrentFillPredictor
        from .modules import ActivationFunction

        self.fill_predictor = GatedRecurrentFillPredictor(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout_rate,
        )

        self.temporal_embed = nn.Linear(1, embedding_dim)
        if self.is_wc or self.is_vrpp:
            self.predict_future = True
        else:
            self.predict_future = False

        self.combine_embeddings = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            ActivationFunction(
                activation_function,
                af_param,
                af_threshold,
                af_replacement_value,
                af_num_params,
                af_uniform_range,
            ),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _get_initial_embeddings(self, nodes):
        # Get the base embeddings from context embedder (without temporal features)
        base_embeddings = self.context_embedder.init_node_embeddings(nodes, temporal_features=False)

        if "fill_history" not in nodes or not self.predict_future:
            return base_embeddings

        fill_history = nodes["fill_history"]
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
        return combined_embeddings

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
        if "fill_history" not in input and self.predict_future:
            batch_size = input["loc"].size(0)
            graph_size = input["loc"].size(1)

            # For VRP-like problems, adjust for depot (excluded from graph size)
            if self.is_vrpp or self.is_wc:
                pass  # graph_size is already correct (num customers)

            fill_history = torch.zeros(
                (batch_size, graph_size, self.temporal_horizon),
                device=input["loc"].device,
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
        if "fill_history" in input and "current_fill" in input:
            input["fill_history"] = self.update_fill_history(input["fill_history"], input["current_fill"])
        return super().compute_simulator_day(input, graph, run_tsp)
