import time

import torch
from tensordict import TensorDict

from logic.src.models.policies.classical.random_local_search import RandomLocalSearchPolicy


def benchmark_ls(batch_size=512, num_nodes=50, iterations=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device} with batch_size={batch_size}, num_nodes={num_nodes}, iterations={iterations}")

    td = TensorDict(
        {
            "locs": torch.rand(batch_size, num_nodes, 2, device=device),
            "demand": torch.rand(batch_size, num_nodes, device=device),
            "capacity": torch.ones(batch_size, device=device),
        },
        batch_size=[batch_size],
    )

    policy = RandomLocalSearchPolicy(env_name="cvrpp", n_iterations=iterations).to(device)

    class MockEnv:
        prize_weight = 1.0
        cost_weight = 1.0

    # Warmup
    _ = policy(td, env=MockEnv())

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    _ = policy(td, env=MockEnv())
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    print(f"Time: {end - start:.4f}s")
    print(f"Time per instance: {(end - start) / batch_size:.6f}s")


if __name__ == "__main__":
    benchmark_ls()
