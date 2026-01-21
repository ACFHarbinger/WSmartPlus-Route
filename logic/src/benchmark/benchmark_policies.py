import time

import pandas as pd
import torch
from tabulate import tabulate

from logic.src.cli.train_lightning import create_model
from logic.src.configs import Config, EnvConfig, ModelConfig, TrainConfig


def benchmark(problem="vrpp", sizes=[20, 50], num_instances=16):
    results = []

    print(f"\nBenchmarking on {num_instances} instances per problem size...")
    print("Device: cuda if available else cpu")

    for graph_size in sizes:
        print(f"\n--- Problem Size: {graph_size} ---")

        # Prepare environment config
        cfg = Config()
        cfg.env = EnvConfig(name=problem, num_loc=graph_size)
        cfg.train = TrainConfig(batch_size=num_instances)

        # Generate generic batch
        # We create a dummy model just to get the env and generator
        dummy_model = create_model(cfg)
        env = dummy_model.env
        td = env.reset(batch_size=[num_instances])

        policies = ["am", "alns", "hgs", "hybrid"]

        for p_name in policies:
            # Configure model
            cfg.model = ModelConfig(name=p_name)

            try:
                model = create_model(cfg)
                model.eval()

                # Warmup
                if p_name == "am":
                    with torch.no_grad():
                        model.policy(td.clone(), env)

                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    out = model.policy(td.clone(), env)
                end_time = time.time()

                duration = end_time - start_time
                avg_time = duration / num_instances

                rewards = out["reward"]
                avg_reward = rewards.mean().item()

                results.append(
                    {
                        "Policy": p_name.upper(),
                        "Size": graph_size,
                        "Avg Time (s)": f"{avg_time:.4f}",
                        "Avg Reward": f"{avg_reward:.2f}",
                        "Total Time (s)": f"{duration:.2f}",
                    }
                )
                print(f"  {p_name.upper()}: {avg_time:.4f}s/inst, Reward: {avg_reward:.2f}")

            except Exception as e:
                print(f"  {p_name.upper()} Failed: {e}")
                results.append(
                    {
                        "Policy": p_name.upper(),
                        "Size": graph_size,
                        "Avg Time (s)": "FAIL",
                        "Avg Reward": "FAIL",
                        "Total Time (s)": "-",
                    }
                )

    df = pd.DataFrame(results)
    print("\n\n=== Final Benchmark Results ===")
    print(tabulate(df, headers="keys", tablefmt="markdown", showindex=False))


if __name__ == "__main__":
    benchmark()
