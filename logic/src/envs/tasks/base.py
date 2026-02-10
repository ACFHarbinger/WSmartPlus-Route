"""
Base problem definition and legacy constants.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.utils.decoding import beam_search as beam_search_func


class BaseProblem:
    """
    Legacy base class for routing problems.
    """

    NAME: str = "base"

    @staticmethod
    def validate_tours(pi: torch.Tensor) -> bool:
        """Validates tours (no duplicates except depot)."""
        if pi.size(-1) <= 1:
            return True
        sorted_pi: torch.Tensor = pi.data.sort(1)[0]
        if not ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all():
            raise ValueError("Tour validation failed: duplicates detected (excluding depot).")
        return True

    @staticmethod
    def get_tour_length(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates tour length."""
        if pi.size(-1) <= 1:
            return torch.zeros(pi.size(0), device=pi.device)

        use_dist_matrix = dist_matrix is not None and isinstance(dist_matrix, torch.Tensor)
        if use_dist_matrix:
            # Simple distance matrix lookup
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask: torch.Tensor = dst_vertices != 0
            pair_mask: torch.Tensor = (src_vertices != 0) & (dst_mask)
            dists: torch.Tensor = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst: torch.Tensor = torch.max(
                dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
                dim=1,
            ).indices
            length: torch.Tensor = (
                dist_matrix[
                    0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0
                ]
                + dists.sum(dim=1)
                + dist_matrix[0, 0, pi[:, 0]]
            )
        else:
            loc_val = dataset.get("locs") if "locs" in dataset else dataset.get("loc")
            waste_val = dataset.get("waste")
            if loc_val is not None and loc_val.size(1) == dataset["depot"].size(0) + (
                waste_val.size(1) if waste_val is not None else 0
            ):
                # already concatenated
                loc_with_depot: Any = loc_val
            elif loc_val is not None:
                loc_with_depot = torch.cat((dataset["depot"][:, None, :], loc_val), 1)
            else:
                # Fallback for empty/missing loc
                return torch.zeros(pi.size(0), device=pi.device)

            d: torch.Tensor = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)
                + (d[:, 0] - dataset["depot"]).norm(p=2, dim=-1)
                + (d[:, -1] - dataset["depot"]).norm(p=2, dim=-1)
            )
        return length

    @classmethod
    def beam_search(cls, input, beam_size, cost_weights, model=None, **kwargs):
        """Beam search bridge."""
        assert model is not None
        fixed = model.precompute_fixed(input, edges=input.get("edges"))

        def propose_expansions(beam):
            """Propose expansions for the current beam search state."""
            if model is None:
                raise ValueError("Model is required for proposing expansions.")
            return model.propose_expansions(beam, fixed, normalize=True)

        # Note: make_state is problem-specific, must be implemented by subclasses
        state = cls.make_state(input, cost_weights=cost_weights, **kwargs)
        return beam_search_func(state, beam_size, propose_expansions)

    @classmethod
    def make_state(
        cls, input_data: Any, edges: Any = None, cost_weights: Any = None, dist_matrix: Any = None, **kwargs: Any
    ) -> Any:
        """
        Bridge to RL4CO environments.
        Initializes a TensorDict from the input and returns a state wrapper.
        """
        from logic.src.envs import get_env
        from logic.src.utils.data.td_state_wrapper import TensorDictStateWrapper

        env_name = cls.NAME

        if isinstance(input_data, dict):
            # Determine batch size from typical batched tensors
            bs = 1
            device = torch.device("cpu")
            for k in ["loc", "locs", "waste"]:
                if k in input_data and torch.is_tensor(input_data[k]):
                    bs = input_data[k].size(0)
                    device = input_data[k].device
                    break

            # Initialize environment (lazy loading or re-use could be better but this is safe)
            env = get_env(env_name, batch_size=torch.Size([bs]), device=device)

            # Create TensorDict, unsqueezing non-batched tensors if needed
            td_data = {}
            for k, v in input_data.items():
                if torch.is_tensor(v):
                    # Key mapping for simulator compatibility
                    target_key = k
                    if k == "loc":
                        target_key = "locs"
                    # Map prize and demand to waste
                    if k in ["prize", "demand"]:
                        target_key = "waste"

                    if v.dim() >= 1 and v.size(0) == bs:
                        td_data[target_key] = v
                    elif v.dim() >= 2:
                        td_data[target_key] = v.unsqueeze(0).expand(bs, *([-1] * v.dim()))
                    else:
                        td_data[target_key] = (
                            v.expand(bs, *([-1] * v.dim())) if v.dim() > 0 else v.unsqueeze(0).expand(bs)
                        )
                else:
                    td_data[k] = v

            td = TensorDict(td_data, batch_size=[bs], device=device)
        elif isinstance(input_data, TensorDict):
            td = input_data
            bs = td.batch_size[0] if len(td.batch_size) > 0 else 1
            env = get_env(env_name, batch_size=torch.Size([bs]), device=td.device)
        else:
            td = TensorDict({}, batch_size=[1])
            env = get_env(env_name, batch_size=torch.Size([1]))

        # Ensure 'dist' and 'edges' are present
        if "dist" not in td.keys() and dist_matrix is not None:
            if dist_matrix.dim() == 2:
                td["dist"] = dist_matrix.unsqueeze(0).expand(td.batch_size[0], -1, -1)
            else:
                td["dist"] = dist_matrix
        if "edges" not in td.keys() and edges is not None:
            if edges.dim() == 2:
                td["edges"] = edges.unsqueeze(0).expand(td.batch_size[0], -1, -1)
            else:
                td["edges"] = edges

        # Consolidate 'locs' logic:
        # We NO LONGER concatenate depot and locs here, as modern envs handle it in reset() or step()
        # and AttentionModel's context embedder also handles separate depot/locs.
        if "locs" not in td.keys() and "loc" in td.keys():
            td["locs"] = td["loc"]

        # Final check for environment-specific required keys
        # Handle must_go mask for selective routing
        if "must_go" in td.keys():
            pass

        # Ensure capacity is present
        if "capacity" not in td.keys():
            profit_vars = kwargs.get("profit_vars")
            if profit_vars and "vehicle_capacity" in profit_vars:
                td["capacity"] = torch.full((td.batch_size[0],), profit_vars["vehicle_capacity"], device=td.device)
            elif "vehicle_capacity" in kwargs:
                td["capacity"] = torch.full((td.batch_size[0],), kwargs["vehicle_capacity"], device=td.device)
            elif env_name in ["wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"]:
                td["capacity"] = torch.ones(td.batch_size[0], device=td.device)

        td_reset = TensorDict(
            source={k: v for k, v in td.items()},
            batch_size=td.batch_size,
            device=td.device,
        )
        td = env.reset(td_reset)

        return TensorDictStateWrapper(td, env_name, env=env)
