import torch
import pytest
from tensordict import TensorDict
from logic.src.models.policies import VectorizedAHVPL

@pytest.mark.parametrize("batch_size", [1, 2])
def test_ahvpl_forward(batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = 11  # 1 depot + 10 bins

    # Mock TensorDict
    td = TensorDict({
        "locs": torch.rand(batch_size, num_nodes, 2, device=device),
        "waste": torch.rand(batch_size, num_nodes - 1, device=device) * 50,
        "capacity": torch.full((batch_size,), 100.0, device=device)
    }, batch_size=[batch_size], device=device)

    policy = VectorizedAHVPL(
        env_name="vrpp",
        n_teams=4,
        max_iterations=2,
        aco_iterations=1,
        alns_iterations=10,
        time_limit=10.0
    ).to(device)

    out = policy(td)

    assert "actions" in out
    assert "reward" in out
    assert out["actions"].shape[0] == batch_size
    assert out["reward"].shape[0] == batch_size

    # Check if actions are within range
    assert (out["actions"] >= 0).all()
    assert (out["actions"] < num_nodes).all()
