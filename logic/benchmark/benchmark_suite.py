import time

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.envs.problems import VRPP
from logic.src.models.attention_model import AttentionModel
from logic.src.models.subnets.factories import AttentionComponentFactory
from logic.src.models.policies.random_local_search import RandomLocalSearchPolicy
from logic.src.policies.adapters.policy_vrpp import run_vrpp_optimizer


def get_dummy_model(device="cpu"):
    factory = AttentionComponentFactory()
    model = AttentionModel(
        embed_dim=128,
        hidden_dim=128,
        problem=VRPP,
        component_factory=factory,
        n_heads=8,
        n_encode_layers=3,
        normalization="instance",
    ).to(device)
    model.eval()
    return model


def benchmark_neural_latency(device="cpu"):
    print(f"\n--- Neural Model Latency (Device: {device}) ---")
    model = get_dummy_model(device)
    batch_sizes = [1, 64, 256]
    n_nodes = 50

    for bs in batch_sizes:
        td = TensorDict(
            {
                "locs": torch.rand(bs, n_nodes, 2, device=device),
                "depot": torch.rand(bs, 2, device=device),
                "waste": torch.rand(bs, n_nodes, device=device),
                "prize": torch.rand(bs, n_nodes, device=device),
                "capacity": torch.ones(bs, device=device),
                "max_waste": torch.ones(bs, device=device),
            },
            batch_size=[bs],
            device=device,
        )

        # Warmup
        model.set_strategy("greedy")
        with torch.no_grad():
            _ = model(td.clone())

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(td.clone())
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / 10
        print(f"Batch Size {bs:3d} | Latency: {avg_time*1000:7.2f} ms | Throughput: {bs/avg_time:10.2f} inst/s")


def benchmark_solvers():
    print("\n--- OR Solver Performance (N=20) ---")
    n_bins = 20
    dist_matrix = np.random.rand(n_bins + 1, n_bins + 1).tolist()
    bins = np.random.rand(n_bins) * 100
    values = {"Q": 100.0, "R": 1.0, "B": 1.0, "C": 0.1, "V": 1.0, "Omega": 0.1, "delta": 0.0, "psi": 0.8}
    binsids = list(range(n_bins + 1))
    must_go = [5, 10, 15]

    for backend in ["gurobi", "hexaly"]:
        start = time.time()
        run_vrpp_optimizer(
            bins=bins,
            distance_matrix=dist_matrix,
            param=0.0,
            media=np.zeros(n_bins),
            desviopadrao=np.zeros(n_bins),
            values=values,
            binsids=binsids,
            must_go=must_go,
            optimizer=backend,
            time_limit=10,
        )
        end = time.time()
        print(f"{backend:8s} | Solve Time: {end - start:7.4f} s")


def benchmark_ls_throughput(device="cpu"):
    print(f"\n--- Local Search Throughput (Device: {device}) ---")
    bs = 512
    n_nodes = 50
    td = TensorDict(
        {
            "locs": torch.rand(bs, n_nodes, 2, device=device),
            "demand": torch.rand(bs, n_nodes, device=device),
            "capacity": torch.ones(bs, device=device),
        },
        batch_size=[bs],
    )

    policy = RandomLocalSearchPolicy(env_name="cvrpp", n_iterations=100).to(device)

    class MockEnv:
        prize_weight = 1.0
        cost_weight = 1.0

    # Warmup
    _ = policy(td, env=MockEnv())

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    _ = policy(td, env=MockEnv())
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    print(f"Batch Size {bs:3d} | Total Time: {end - start:7.4f} s | Throughput: {bs/(end-start):10.2f} inst/s")



def run_benchmarks(args):
    """
    Main entry point for the benchmark suite.
    """
    subset = args.get("subset", "all")
    device_arg = args.get("device", "auto")
    output_file = args.get("output")

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    results = {}

    print(f"Running Benchmarks (Subset: {subset}, Device: {device})...")

    if subset in ["all", "neural"]:
        # Capture stdout? Or just run and let it print?
        # For now, let it print.
        benchmark_neural_latency(device)

    if subset in ["all", "ls"]:
        benchmark_ls_throughput(device)

    if subset in ["all", "solvers"]:
        benchmark_solvers()

    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Simulate args for direct execution
    run_benchmarks({"subset": "all", "device": "auto"})
