
import pytest
import torch
from tensordict import TensorDict
from logic.src.envs.pdp import PDPEnv
from logic.src.envs.generators import PDPGenerator

class TestPDPEnv:
    @pytest.fixture
    def env(self):
        # Small env: 5 pairs = 10 nodes + depot = 11 nodes
        return PDPEnv(num_loc=5, device="cpu")

    def test_generator(self, env):
        td = env.reset(batch_size=[2])
        # Generator makes 2*N locs
        assert td["locs"].shape == (2, 10, 2)
        assert td["depot"].shape == (2, 2)
        assert td["current_load"].shape == (2, 1)
        assert td["visited"].shape == (2, 11)

    def test_mask_precedence(self, env):
        td = env.reset(batch_size=[1])
        mask = env._get_action_mask(td)

        # 0: Depot (already visited -> should be False initially until all done?
        # Actually standard logic makes it False until all nodes visited)
        assert mask[0, 0] == False

        # 1..5: Pickups (should be True)
        assert mask[0, 1:6].all()

        # 6..10: Deliveries (should be False because pickups not visited)
        assert not mask[0, 6:11].any()

    def test_step_logic(self, env):
        td = env.reset(batch_size=[1])

        # Action: Visit Pickup 1 (index 1)
        action = torch.tensor([1], dtype=torch.long)
        td["action"] = action
        td = env.step(td)["next"]

        # Check load increased
        assert td["current_load"][0].item() == 1.0

        # Check Pickup 1 visited
        assert td["visited"][0, 1] == True

        # Check Delivery 1 (index 1+5=6) is now valid
        mask = env._get_action_mask(td)
        assert mask[0, 6] == True

        # Action: Visit Delivery 1 (index 6)
        action = torch.tensor([6], dtype=torch.long)
        td["action"] = action
        td = env.step(td)["next"]

        # Check load decreased
        assert td["current_load"][0].item() == 0.0

        # Check Delivery 1 visited
        assert td["visited"][0, 6] == True

    def test_capacity_constraint(self, env):
        # Force a small capacity
        td = env.reset(batch_size=[1])
        td["capacity"] = torch.tensor([1.0]) # Capacity 1

        # Pick up item 1
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        # Load is 1.0. Capacity is 1.0.
        assert td["current_load"][0].item() == 1.0

        # Now masking should forbid other pickups (indices 2..5)
        mask = env._get_action_mask(td)
        # Pickups 2..5 should be False
        assert not mask[0, 2:6].any()

        # Delivery 1 (index 6) should be True
        assert mask[0, 6] == True
