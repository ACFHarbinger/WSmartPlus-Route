from unittest.mock import MagicMock

import pytest
import torch
from logic.src.pipeline.rl.common.time_training import TimeBasedMixin, prepare_time_dataset
from tensordict import TensorDict


class MockTrainer(TimeBasedMixin):
    pass


class TestTimeTraining:
    @pytest.fixture
    def trainer(self):
        t = MockTrainer()
        t.setup_time_training({"temporal_horizon": 5})
        return t

    def test_setup_time_training(self):
        t = MockTrainer()
        t.setup_time_training({"temporal_horizon": 10})
        assert t.temporal_horizon == 10
        assert t.current_day == 0
        assert t.fill_history == []

    def test_update_dataset_for_day(self, trainer):
        # Mock dataset and TensorDict
        batch_size = 2
        num_nodes = 3  # + 1 depot = 4 cols

        # Initial Demand:
        # B0: [0, 10, 10, 10]
        # B1: [0, 20, 20, 20]
        demand = torch.tensor([[0.0, 10.0, 10.0, 10.0], [0.0, 20.0, 20.0, 20.0]])

        # Locs needed for num_nodes calculation logic
        # shape [B, N+1, 2]
        locs = torch.zeros(batch_size, num_nodes + 1, 2)

        td = TensorDict(
            {
                "demand": demand,
                "locs": locs,
                "capacity": torch.tensor([100.0, 100.0]),
                "generation_rate": torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 2.0, 2.0, 2.0]]),  # Explicit rate
            },
            batch_size=[batch_size],
        )

        dataset = MagicMock()
        dataset.data = td
        trainer.train_dataset = dataset

        # Routes / Actions
        # B0 visited 1, 2. (Indices)
        # B1 visited 3.
        # Actions shape [batch, seq_len]
        routes = torch.tensor([[1, 2, 0], [3, 0, 0]])

        trainer.update_dataset_for_day(routes, day=0)

        assert trainer.current_day == 0
        assert len(trainer.fill_history) == 1

        # Check logic:
        # 1. Process Collections:
        # B0: visited 1, 2. Demand at 1, 2 becomes 0.
        # Node 3 stays 10.
        # B1: visited 3. Demand at 3 becomes 0.
        # Node 1, 2 stay 20.

        # 2. Accumulation (Noise * Rate)
        # We can't predict exact noise, but min increase is 0 if clamped?
        # Noise is N(1, 0.1) * Rate.
        # Rate is 1 (B0), 2 (B1).
        # New demand > 0.

        new_demand = td["demand"]

        # B0 Node 1: Was 10 -> Reset to 0 -> Add Accumulation (~1.0)
        # Should be around 1.0
        assert new_demand[0, 1] > 0.5 and new_demand[0, 1] < 2.0

        # B0 Node 3: Was 10 -> Not visited -> Add Accumulation (~1.0). Total ~11.
        assert new_demand[0, 3] > 10.0

        # B1 Node 3: Was 20 -> Reset 0 -> Add Accum (~2.0)
        assert new_demand[1, 3] > 1.0

    def test_prepare_time_dataset(self):
        td = TensorDict({}, batch_size=[1])
        dataset = MagicMock()
        dataset.data = td

        prepare_time_dataset(dataset, day=5, history=[])
        assert td["current_day"].item() == 5
