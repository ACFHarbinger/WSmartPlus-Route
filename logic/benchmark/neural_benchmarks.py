"""
Neural model performance benchmarks.

Measures latency and throughput (instances/sec) for Attention models (AM, TAM)
across different graph sizes and batch sizes.
"""

import time
from typing import Any, Dict, List

import torch
from tensordict import TensorDict

from logic.src.models.policies.am import AttentionModelPolicy


def benchmark_neural_model(
    model_name: str = "am",
    num_nodes: int = 50,
    batch_size: int = 128,
    decode_type: str = "greedy",
) -> Dict[str, float]:
    """
    Benchmark a neural model's inference performance.

    Args:
        model_name: Name of the model to benchmark.
        num_nodes: Number of nodes in the graph.
        batch_size: Inference batch size.
        decode_type: 'greedy' or 'sampling'.

    Returns:
        Dict with latency and throughput metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Benchmarking {model_name.upper()} (nodes={num_nodes}, batch={batch_size}, device={device})...")

    # Setup environment
    from logic.src.envs import get_env
    env = get_env("vrpp", device=device)

    # Instantiate policy directly
    policy = AttentionModelPolicy(
        env_name="vrpp",
        embed_dim=128,
        hidden_dim=128,
        n_encode_layers=3,
        n_heads=8,
    ).to(device)
    policy.eval()

    # Generate dummy data using env's generator
    td = env.generator(batch_size=(batch_size,))
    td = env.reset(td)

    # Warmup
    with torch.no_grad():
        _ = policy(td, env=env, decode_type=decode_type)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measurement
    start = time.time()
    with torch.no_grad():
        _ = policy(td, env=env, decode_type=decode_type)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    latency = end - start
    instances_per_second = batch_size / latency
    ms_per_instance = (latency / batch_size) * 1000

    print(f"    - Latency: {latency:.4f}s")
    print(f"    - Throughput: {instances_per_second:.2f} instances/s")
    print(f"    - Latency per Instance: {ms_per_instance:.4f}ms")

    return {
        "latency": latency,
        "throughput": instances_per_second,
        "ms_per_instance": ms_per_instance,
    }


def run_full_neural_suite():
    """Run benchmarks across various configurations."""
    results: List[Dict[str, Any]] = []

    graph_sizes = [20, 50, 100]
    batch_sizes = [64, 256]

    for size in graph_sizes:
        for batch in batch_sizes:
            res = benchmark_neural_model(model_name="am", num_nodes=size, batch_size=batch)
            # Use type-any dict for results to avoid mapping update errors
            entry: Dict[str, Any] = {**res}
            entry.update({"model": "am", "nodes": size, "batch": batch})
            results.append(entry)
            print("-" * 20)

    return results


if __name__ == "__main__":
    run_full_neural_suite()
