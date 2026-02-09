from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.common.autoregressive_policy import AutoregressivePolicy
from logic.src.models.common.improvement_decoder import ImprovementDecoder
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder

if TYPE_CHECKING:
    pass


class ImprovementStepDecoder(ImprovementDecoder):
    """
    Decoder that selects a vectorized operator to improve the solution.
    Output is a probability distribution over available operators.
    """

    def __init__(self, embed_dim: int = 128, n_operators: int = 6, hidden_dim: int = 128):
        super().__init__(embed_dim=embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_operators)
        )
        self.context_projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict operator selection logits.

        Args:
            td: TensorDict containing current state/solution.
            embeddings: Graph embeddings [Batch, Nodes, Embed].
            env: Environment.

        Returns:
            Tuple of (logits, action_mask).
        """
        # Simple Global Context: Mean pooling of node embeddings
        # Real implementation might encode current tour state more deeply
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]

        graph_context = embeddings.mean(dim=1)  # [B, Embed]

        # Project and predict
        context = self.context_projection(graph_context)
        logits = self.output_head(context)  # [B, n_operators]

        # For now, no mask (all operators always available)
        mask = torch.zeros_like(logits, dtype=torch.bool)

        return logits, mask


class HybridTwoStagePolicy(AutoregressivePolicy):
    """
    Hybrid Policy with 2-Stage Decoder:
    1. Initialization: Select HGS, ALNS, or ACO.
    2. Refinement: Iteratively select vectorized operators.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        refine_steps: int = 10,
        **kwargs,
    ):
        super().__init__(env_name=env_name, embed_dim=embed_dim)

        self.refine_steps = refine_steps

        self.init_embedding = get_init_embedding(env_name, embed_dim)
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads, embed_dim=embed_dim, feed_forward_hidden=hidden_dim, n_layers=n_encode_layers, **kwargs
        )

        # Stage 1: Init Selector (3 choices: HGS, ALNS, ACO)
        self.init_router = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.INIT_ALGOS = ["hgs", "alns", "aco"]

        # Stage 2: Improvement Decoder
        # Import Vectorized Operators locally to avoid circular imports
        from logic.src.models.policies.operators import (
            # Destroy
            vectorized_cluster_removal,
            # Exchange
            vectorized_cross_exchange,
            vectorized_ejection_chain,
            # Repair
            vectorized_greedy_insertion,
            vectorized_lambda_interchange,
            # Route
            vectorized_lkh,
            vectorized_or_opt,
            vectorized_random_removal,
            vectorized_regret_k_insertion,
            # Move
            vectorized_relocate,
            vectorized_shaw_removal,
            vectorized_string_removal,
            vectorized_swap,
            vectorized_swap_star,
            vectorized_three_opt,
            vectorized_two_opt,
            vectorized_two_opt_star,
            # Unstringing
            vectorized_type_i_unstringing,
            vectorized_type_ii_unstringing,
            vectorized_type_iii_unstringing,
            vectorized_type_iv_unstringing,
            vectorized_worst_removal,
        )

        # Available Operators
        self.OPERATORS = [
            # Destroy
            vectorized_cluster_removal,
            vectorized_random_removal,
            vectorized_shaw_removal,
            vectorized_string_removal,
            vectorized_worst_removal,
            # Exchange
            vectorized_cross_exchange,
            vectorized_ejection_chain,
            vectorized_lambda_interchange,
            vectorized_or_opt,
            # Move
            vectorized_relocate,
            vectorized_swap,
            # Repair
            vectorized_greedy_insertion,
            vectorized_regret_k_insertion,
            # Route
            vectorized_lkh,
            vectorized_swap_star,
            vectorized_three_opt,
            vectorized_two_opt,
            vectorized_two_opt_star,
            # Unstringing
            vectorized_type_i_unstringing,
            vectorized_type_ii_unstringing,
            vectorized_type_iii_unstringing,
            vectorized_type_iv_unstringing,
        ]
        self.improvement_decoder = ImprovementStepDecoder(
            embed_dim=embed_dim, n_operators=len(self.OPERATORS), hidden_dim=hidden_dim
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "greedy",
        **kwargs,
    ) -> Dict[str, Any]:
        # Import locally to avoid circular imports
        from logic.src.models.policies.adaptive_large_neighborhood_search import VectorizedALNS
        from logic.src.models.policies.ant_colony_system import VectorizedACOPolicy
        from logic.src.models.policies.hybrid_genetic_search import VectorizedHGS

        # Instantiate ACO model here if needed, or keep it lazy
        if not hasattr(self, "aco_model"):
            self.aco_model = VectorizedACOPolicy(env_name=self.env_name, embed_dim=self.encoder.embed_dim, **kwargs)

        batch_size = td.size(0)
        device = td["locs"].device

        # 1. Encode Graph
        init_embeds = self.init_embedding(td)
        embeddings = self.encoder(init_embeds)
        graph_context = embeddings.mean(dim=1)

        # 2. Stage 1: Initialization
        init_logits = self.init_router(graph_context)

        if strategy == "greedy":
            init_choice_idx = init_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(init_logits, dim=-1)
            init_choice_idx = torch.multinomial(probs, 1).squeeze(-1)

        # Execute selected initializers
        # Initialize tours container
        tours = torch.zeros(batch_size, td["locs"].size(1), dtype=torch.long, device=device)

        # Group by choice to run batch solvers
        for algo_idx, algo_name in enumerate(self.INIT_ALGOS):
            mask = init_choice_idx == algo_idx
            if not mask.any():
                continue

            sub_td = td[mask]

            # Prepare Solver Inputs
            dist_matrix = self._get_dist_matrix(sub_td)
            demands = sub_td.get("demand", None)

            # Using 1.0 capacity default if not present, as placeholders
            # In real scenario, capacity should come from sub_td
            vehicle_capacity = sub_td.get("vehicle_capacity", 1.0)
            if vehicle_capacity.dim() > 0:
                # If capacity is per-instance, take mean or first (usually homogenous in batch)
                # Solvers might expect scalar or tensor
                pass

            if algo_name == "hgs":
                solver = VectorizedHGS(dist_matrix, demands, vehicle_capacity=1.0, device=device)
                rand_sol = self._get_random_tours(sub_td)
                sub_tours, _ = solver.solve(rand_sol, n_generations=5, population_size=10)

            elif algo_name == "alns":
                solver = VectorizedALNS(dist_matrix, demands, vehicle_capacity=1.0, device=device)
                rand_sol = self._get_random_tours(sub_td)
                sub_tours, _ = solver.solve(rand_sol, n_iterations=10)

            elif algo_name == "aco":
                # VectorizedACOPolicy acts as a model/policy
                out = self.aco_model(sub_td, env, **kwargs)
                sub_tours = out["actions"]

            # Scatter back
            if isinstance(sub_tours, list):
                # Handle list of lists (routes) conversion to padded tensor
                pass  # Skipping complex conversion for brevity in V1
            else:
                tours[mask] = sub_tours.to(dtype=torch.long)

        # 3. Stage 2: Refinement Loop
        current_tours = tours
        dist_matrix_all = self._get_dist_matrix(td)

        # State mechanism for partial solutions (Destroy/Repair)
        # We track removed_nodes for each batch element
        # Initialize with placeholder (no nodes removed)
        # B x 0 tensor
        removed_nodes_state = torch.zeros((batch_size, 0), dtype=torch.long, device=device)

        log_likelihood = 0.0

        for step in range(self.refine_steps):
            # Predict Operator
            op_logits, _ = self.improvement_decoder(td, embeddings, env)

            if strategy == "greedy":
                op_idx = op_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(op_logits, dim=-1)
                op_idx = torch.multinomial(probs, 1).squeeze(-1)
                log_p = torch.log(probs.gather(1, op_idx.unsqueeze(-1)) + 1e-10).squeeze(-1)
                log_likelihood = log_likelihood + log_p

            # Apply Operators (Grouped by op_idx)
            next_tours = current_tours.clone()

            # List to collect new removed states
            updated_removed_nodes_list = [None] * batch_size

            for o_i, operator_fn in enumerate(self.OPERATORS):
                mask = op_idx == o_i
                if not mask.any():
                    continue

                sub_tours = current_tours[mask]
                sub_dist = dist_matrix_all[mask]

                # Get current removed nodes for this group
                if removed_nodes_state.size(0) == batch_size:
                    sub_removed = removed_nodes_state[mask]
                else:
                    # Fallback if state shape mismatch (should not happen)
                    sub_removed = torch.zeros((sub_tours.size(0), 0), dtype=torch.long, device=device)

                # Inspect signature
                import inspect

                sig = inspect.signature(operator_fn)
                valid_kwargs = {}

                # Setup kwargs
                if "max_iterations" in sig.parameters:
                    valid_kwargs["max_iterations"] = 5

                if "n_remove" in sig.parameters:
                    # Default remove count
                    valid_kwargs["n_remove"] = int(sub_tours.size(1) * 0.1) + 1

                if "removed_nodes" in sig.parameters:
                    # This is a Repair operator
                    valid_kwargs["removed_nodes"] = sub_removed

                # Apply
                try:
                    result = operator_fn(sub_tours, sub_dist, **valid_kwargs)
                except Exception:
                    # If operator fails, fallback to identity
                    result = sub_tours

                # Parse result
                if isinstance(result, tuple):
                    new_sub_tours, new_sub_removed = result
                else:
                    new_sub_tours = result
                    if "removed_nodes" in sig.parameters:
                        # Repair -> Cleared
                        new_sub_removed = torch.zeros((sub_tours.size(0), 0), dtype=torch.long, device=device)
                    else:
                        # Move/Improve -> Preserve
                        new_sub_removed = sub_removed

                # Update next_tours
                next_tours[mask] = new_sub_tours

                # Store new removed output
                sub_indices = torch.nonzero(mask).squeeze(-1)
                if sub_indices.dim() == 0:
                    sub_indices = sub_indices.unsqueeze(0)

                for i, batch_idx_k in enumerate(sub_indices):
                    updated_removed_nodes_list[batch_idx_k.item()] = new_sub_removed[i]

            # Re-assemble removed_nodes_state
            max_len = 0
            for r in updated_removed_nodes_list:
                if r is not None:
                    max_len = max(max_len, r.size(0))

            if max_len == 0:
                removed_nodes_state = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
            else:
                new_state = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
                for b in range(batch_size):
                    r = updated_removed_nodes_list[b]
                    if r is not None:
                        l = r.size(0)
                        new_state[b, :l] = r
                removed_nodes_state = new_state

            current_tours = next_tours

        # Calculate final reward
        reward = env.get_reward(td, current_tours)

        return {
            "actions": current_tours,
            "reward": reward,
            "log_likelihood": log_likelihood,
            "init_choice": init_choice_idx,
        }

    def _get_dist_matrix(self, td):
        locs = td["locs"]
        return torch.cdist(locs, locs)

    def _get_random_tours(self, td):
        B, N, _ = td["locs"].shape
        # Simple random permutation for N-1 nodes (excluding depot 0)
        perms = torch.stack([torch.randperm(N - 1) + 1 for _ in range(B)]).to(td.device)
        return perms
