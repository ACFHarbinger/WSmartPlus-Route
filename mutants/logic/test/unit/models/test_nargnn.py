"""Tests for NARGNN model."""

import torch
from tensordict import TensorDict
from unittest.mock import MagicMock

from logic.src.models.nargnn import NARGNN
from logic.src.models.policies.nargnn import NARGNNPolicy


class TestNARGNN:
    """Tests for NARGNN Model and Policy."""

    def setup_method(self):
        self.batch_size = 2
        self.num_nodes = 50   # Increased to avoid k_sparse issues
        self.env = MagicMock()
        self.env.name = "vrpp"

        # Mock env.step to return results matching input batch size
        self.step_count = 0
        def mock_step(td):
            self.step_count += 1
            done = torch.ones(td.batch_size, dtype=torch.bool) if self.step_count >= self.num_nodes else torch.zeros(td.batch_size, dtype=torch.bool)
            return {"next": TensorDict({
                "done": done,
                "mask": torch.ones(td.batch_size[0], self.num_nodes, dtype=torch.bool),
            }, batch_size=td.batch_size)}

        self.env.step.side_effect = mock_step

        # Mock env.get_reward
        self.env.get_reward.side_effect = lambda td, actions: -torch.ones(actions.shape[0])

        self.policy = NARGNNPolicy(
            embed_dim=32,
            env_name="vrpp",
            num_layers_heatmap_generator=2,
            num_layers_graph_encoder=2,
        )

        self.model = NARGNN(
            embed_dim=32,
            env_name="vrpp",
            num_layers_heatmap_generator=2,
            num_layers_graph_encoder=2,
            baseline=None,
        )

    def test_policy_forward(self):
        """Verify NARGNNPolicy forward pass."""
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
            "prize": torch.rand(self.batch_size, self.num_nodes),
            "depot": torch.rand(self.batch_size, 2),
            "current_node": torch.zeros(self.batch_size, dtype=torch.long),
            "mask": torch.ones(self.batch_size, self.num_nodes, dtype=torch.bool),
            "done": torch.zeros(self.batch_size, dtype=torch.bool),
        }, batch_size=[self.batch_size])

        out = self.policy(td, self.env)

        assert "actions" in out
        assert "reward" in out
        assert "log_likelihood" in out

        assert out["actions"].shape == (self.batch_size, self.num_nodes)
        assert out["reward"].shape == (self.batch_size,)

    def test_model_forward_training(self):
        """Verify NARGNN model forward pass for training."""
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
            "prize": torch.rand(self.batch_size, self.num_nodes),
            "depot": torch.rand(self.batch_size, 2),
            "current_node": torch.zeros(self.batch_size, dtype=torch.long),
            "mask": torch.ones(self.batch_size, self.num_nodes, dtype=torch.bool),
            "done": torch.zeros(self.batch_size, dtype=torch.bool),
        }, batch_size=[self.batch_size])

        out = self.model(td, self.env)

        assert "loss" in out
        assert "reward" in out
        assert "baseline" in out
        assert out["loss"].requires_grad
