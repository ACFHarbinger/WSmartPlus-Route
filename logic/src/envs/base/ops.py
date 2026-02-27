"""
Core operations mixin for RL4CO environment.
"""

from abc import abstractmethod
from typing import Any, Optional, cast

import torch
from tensordict import TensorDictBase


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
        from torchrl.data import DiscreteTensorSpec, TensorSpec, UnboundedContinuousTensorSpec

        # self.device and self.batch_size are assumed from EnvBase
        batch_size = getattr(self, "batch_size", torch.Size([]))
        device = getattr(self, "device", "cpu")

        # Cast to TensorSpec to satisfy Pyrefly in cases of multiple inheritance
        # Ensure shape is a torch.Size object
        self.done_spec = cast(
            TensorSpec, DiscreteTensorSpec(n=2, shape=torch.Size((*batch_size, 1)), dtype=torch.bool, device=device)
        )
        self.reward_spec = cast(
            TensorSpec, UnboundedContinuousTensorSpec(shape=torch.Size((*batch_size, 1)), device=device)
        )

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Execute action and update state.
        Synchronizes environment batch size with input TensorDict.
        """
        self.batch_size = tensordict.batch_size
        out = cast(TensorDictBase, super().step(tensordict))  # type: ignore[misc]
        return out

    def reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """
        Sync batch size and initialize state.
        """
        # Support both 'td' and 'tensordict' naming
        if tensordict is None:
            tensordict = kwargs.pop("tensordict", None)
            if tensordict is None:
                tensordict = kwargs.pop("td", None)

        if tensordict is not None:
            self.batch_size = tensordict.batch_size

        return self._reset(tensordict, **kwargs)

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """
        Initialize episode state from problem instance.
        """
        batch_size_arg = kwargs.pop("batch_size", None)
        if tensordict is None:
            if self.generator is None:  # type: ignore[attr-defined]
                raise ValueError("Either provide tensordict or set a generator for the environment")
            tensordict = self.generator(batch_size_arg or self.batch_size)  # type: ignore[attr-defined]
        else:
            tensordict = tensordict.clone()

        assert tensordict is not None
        # Move to device
        tensordict = tensordict.to(self.device)  # type: ignore[attr-defined]

        # Call problem-specific reset (must be implemented by subclasses)
        tensordict = self._reset_instance(tensordict)

        # Add common fields
        tensordict["action_mask"] = self._get_action_mask(tensordict)
        tensordict["i"] = torch.zeros(tensordict.batch_size, dtype=torch.long, device=self.device)  # type: ignore[attr-defined]

        # Initialize done signals with [B, 1] shape
        tensordict["done"] = torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]

        # Safe key check on tensordict
        if "terminated" in tensordict:
            tensordict["terminated"] = torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]
        if "truncated" in tensordict:
            tensordict["truncated"] = torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]

        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Execute action and return new state as 'next' entry.
        """
        # Copy and clean to avoid cycles
        td_next = tensordict.copy()
        for key in ["next", "reward", "done"]:
            if key in td_next.keys():
                del td_next[key]

        # Execute problem-specific step
        td_next = self._step_instance(td_next)

        # Update common fields in the next state
        td_next["i"] = tensordict["i"] + 1
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
        reward = self._get_reward(td_next, tensordict.get("action", None))

        # Ensure reward shape matches batch_size [B, 1] for consistency
        if reward.dim() == len(td_next.batch_size):
            reward = reward.unsqueeze(-1)
        elif reward.dim() > len(td_next.batch_size) + 1:
            reward = reward.reshape((*td_next.batch_size, 1))

        td_next["reward"] = reward

        return td_next

    def make_state(
        self,
        nodes: TensorDictBase,
        edges: Optional[torch.Tensor] = None,
        cost_weights: Optional[torch.Tensor] = None,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a state wrapper for the environment.
        This provides the API expected by the constructive decoders.
        """
        from logic.src.utils.data.td_state_wrapper import TensorDictStateWrapper

        # If nodes is not already initialized (missing 'current_node' etc), reset it
        tensordict = nodes
        if tensordict is not None and "current_node" not in tensordict:
            tensordict = self._reset(tensordict)

        assert tensordict is not None
        # Ensure dist is in tensordict if provided
        if dist_matrix is not None:
            tensordict["dist"] = dist_matrix

        return TensorDictStateWrapper(tensordict, self.name, self)  # type: ignore[attr-defined]

    def get_costs(
        self,
        tensordict: TensorDictBase,
        pi: torch.Tensor,
        cost_weights: Optional[torch.Tensor] = None,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Compute costs for a sequence of actions.
        This provides the API expected by the AttentionModel.
        """
        # Reset and follow pi
        curr_td = self.reset(tensordict)
        for i in range(pi.size(1)):
            curr_td["action"] = pi[:, i]
            # Use tensordict instead of td in step call result
            result = self.step(curr_td)
            curr_td = result["next"]

        reward = self.get_reward(curr_td, pi)

        # Return negative reward as cost (AttentionModel expects minimizing cost)
        # We also return the final td so metrics can be extracted
        return -reward, {"total": -reward}, curr_td

    def _step_instance(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Core state transition logic common to most routing problems.
        Updates visited mask, current node, and tour tracking.
        """
        action = tensordict["action"]
        current = tensordict.get("current_node", torch.zeros_like(action))

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

        locs = tensordict["locs"]

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        tensordict["tour_length"] = tensordict.get("tour_length", torch.zeros_like(distance)) + distance

        # Update visited
        tensordict["visited"] = tensordict["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        tensordict["current_node"] = action.unsqueeze(-1)

        # Append to tour
        assert tensordict is not None
        if "tour" not in tensordict:
            tensordict["tour"] = action.unsqueeze(-1)
        else:
            tensordict["tour"] = torch.cat([tensordict["tour"], action.unsqueeze(-1)], dim=-1)

        return tensordict

    def _check_done(self, tensordict: TensorDictBase) -> torch.Tensor:
        """
        Check if episodes are complete.
        Returns: [B] bool tensor
        """
        current_node = tensordict.get("current_node", None)
        step_count = tensordict.get("i", None)

        if current_node is None or step_count is None:
            return torch.zeros(tensordict.batch_size, dtype=torch.bool, device=tensordict.device)

        # Done if we're back at depot (node 0) after at least one step
        # Handle current_node being [B] or [B, 1]
        node = current_node.squeeze(-1) if current_node.dim() > 1 else current_node
        steps = step_count.squeeze(-1) if step_count.dim() > 1 else step_count
        done = (node == 0) & (steps > 0)
        try:
            return done.reshape(tensordict.batch_size)
        except Exception:
            return done.flatten().reshape(tensordict.batch_size)

    def get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Public method to compute rewards.
        """
        return self._get_reward(tensordict, actions)

    @abstractmethod
    def _reset_instance(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Problem-specific instance initialization."""
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward."""
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Return mask of valid actions."""
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]):
        """Set random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
