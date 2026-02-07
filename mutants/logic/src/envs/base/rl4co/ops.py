"""
Core operations mixin for RL4CO environment.
"""

from abc import abstractmethod
from typing import Any, Optional, cast

import torch
from tensordict import TensorDict


class OpsMixin:
    """
    Mixin to handle Step, Reset, and Cost calculations.
    """

    def _make_spec(self, generator: Optional[Any] = None) -> None:
        """
        Create environment specs (reward_spec, done_spec, etc.).

        Args:
            generator: Optional data generator.
        """
        from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec

        # self.device is assumed from EnvBase
        self.done_spec = DiscreteTensorSpec(n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1), device=self.device)

    def step(self, td: TensorDict) -> TensorDict:
        """
        Execute action and update state.
        Synchronizes environment batch size with input TensorDict.
        """
        self.batch_size = td.batch_size
        out = cast(TensorDict, super().step(td))
        return out

    def reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """
        Sync batch size and initialize state.
        """
        # Support both 'td' and 'tensordict' naming
        if td is None and "tensordict" in kwargs:
            td = kwargs.pop("tensordict")

        if td is not None:
            self.batch_size = td.batch_size

        return self._reset(td, **kwargs)

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[int] = None) -> TensorDict:
        """
        Initialize episode state from problem instance.
        """
        if td is None:
            if self.generator is None:
                raise ValueError("Either provide td or set a generator for the environment")
            td = self.generator(batch_size or self.batch_size)
        else:
            td = td.clone()

        # Move to device
        td = td.to(self.device)

        # Call problem-specific reset (must be implemented by subclasses)
        td = self._reset_instance(td)

        # Add common fields
        td["action_mask"] = self._get_action_mask(td)
        td["i"] = torch.zeros(td.batch_size, dtype=torch.long, device=self.device)

        # Initialize done signals with [B, 1] shape
        td["done"] = torch.zeros((*td.batch_size, 1), dtype=torch.bool, device=self.device)
        if "terminated" in td.keys():
            td["terminated"] = torch.zeros((*td.batch_size, 1), dtype=torch.bool, device=self.device)
        if "truncated" in td.keys():
            td["truncated"] = torch.zeros((*td.batch_size, 1), dtype=torch.bool, device=self.device)

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Execute action and return new state as 'next' entry.
        """
        # Copy and clean to avoid cycles
        td_next = td.copy()
        for key in ["next", "reward", "done"]:
            if key in td_next.keys():
                del td_next[key]

        # Execute problem-specific step
        td_next = self._step_instance(td_next)

        # Update common fields in the next state
        td_next["i"] = td["i"] + 1
        td_next["action_mask"] = self._get_action_mask(td_next)
        done = self._check_done(td_next)

        # Ensure done shape matches spec [B, 1]
        if done.dim() == len(td_next.batch_size):
            done = done.unsqueeze(-1)
        elif done.dim() > len(td_next.batch_size) + 1:
            done = done.reshape((*td_next.batch_size, 1))

        td_next["done"] = done
        td_next["terminated"] = done.clone()
        td_next["truncated"] = torch.zeros_like(done)

        # Reward calculation
        reward = self._get_reward(td_next, td.get("action", None))

        # Ensure reward shape matches batch_size [B, 1] for consistency
        if reward.dim() == len(td_next.batch_size):
            reward = reward.unsqueeze(-1)
        elif reward.dim() > len(td_next.batch_size) + 1:
            reward = reward.reshape((*td_next.batch_size, 1))

        td_next["reward"] = reward

        return td_next

    def make_state(
        self,
        nodes: TensorDict,
        edges: Optional[torch.Tensor] = None,
        cost_weights: Optional[torch.Tensor] = None,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a state wrapper for the environment.
        This provides the API expected by the constructive decoders.
        """
        from logic.src.utils.data.td_utils import TensorDictStateWrapper

        # If nodes is not already initialized (missing 'current_node' etc), reset it
        td = nodes
        if "current_node" not in td.keys():
            td = self._reset(td)

        # Ensure dist is in td if provided
        if dist_matrix is not None:
            td["dist"] = dist_matrix

        return TensorDictStateWrapper(td, self.name, self)

    def get_costs(
        self,
        td: TensorDict,
        pi: torch.Tensor,
        cost_weights: Optional[torch.Tensor] = None,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Compute costs for a sequence of actions.
        This provides the API expected by the AttentionModel.
        """
        # Reset and follow pi
        curr_td = self.reset(td)
        for i in range(pi.size(1)):
            curr_td["action"] = pi[:, i]
            curr_td = self.step(curr_td)["next"]

        reward = self.get_reward(curr_td, pi)

        # Return negative reward as cost (AttentionModel expects minimizing cost)
        return -reward, {"total": -reward}, None

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Core state transition logic common to most routing problems.
        Updates visited mask, current node, and tour tracking.
        """
        action = td["action"]
        current = td.get("current_node", torch.zeros_like(action))

        # Robustly squeeze to [B]
        if current.dim() > 1:
            current = current.squeeze(-1)
        if action.dim() > 1:
            action = action.squeeze(-1)
        # Ensure they are at least 1D for slicing
        if current.dim() == 0:
            current = current.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        locs = td["locs"]

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        td["tour_length"] = td.get("tour_length", torch.zeros_like(distance)) + distance

        # Update visited
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        td["current_node"] = action.unsqueeze(-1)

        # Append to tour
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """
        Check if episodes are complete.
        Returns: [B] bool tensor
        """
        current_node = td.get("current_node", None)
        step_count = td.get("i", None)

        if current_node is None or step_count is None:
            return torch.zeros(td.batch_size, dtype=torch.bool, device=td.device)

        # Done if we're back at depot (node 0) after at least one step
        # Handle current_node being [B] or [B, 1]
        node = current_node.squeeze(-1) if current_node.dim() > 1 else current_node
        steps = step_count.squeeze(-1) if step_count.dim() > 1 else step_count
        done = (node == 0) & (steps > 0)
        try:
            return done.reshape(td.batch_size)
        except Exception:
            return done.flatten().reshape(td.batch_size)

    def get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Public method to compute rewards.
        """
        return self._get_reward(td, actions)

    @abstractmethod
    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Problem-specific instance initialization."""
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward."""
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Return mask of valid actions."""
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]):
        """Set random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
