import torch
from tensordict import TensorDict
from logic.src.models.policies.hybrid_volleyball_premier_league import VectorizedHVPL

def test_vectorized_hvpl():
    batch_size = 2
    num_nodes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup mock data
    td = TensorDict({
        "locs": torch.rand(batch_size, num_nodes, 2),
        "waste": torch.rand(batch_size, num_nodes - 1),
        "capacity": torch.full((batch_size,), 10.0)
    }, batch_size=[batch_size]).to(device)

    # 2. Initialize policy
    policy = VectorizedHVPL(
        env_name="cvrp",
        n_teams=3,
        max_iterations=2,
        alns_iterations=10,
        time_limit=5.0
    ).to(device)

    # 3. Forward pass
    out = policy(td)

    # 4. Verify output structure
    assert "actions" in out
    assert "reward" in out
    assert "cost" in out

    assert out["actions"].dim() == 2
    assert out["actions"].shape[0] == batch_size
    assert out["reward"].shape[0] == batch_size
    assert out["cost"].shape[0] == batch_size
