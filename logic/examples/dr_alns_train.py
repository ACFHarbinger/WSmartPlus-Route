"""
Example training script for DR-ALNS.

Demonstrates how to train a PPO agent to control ALNS parameters online.

Usage:
    python -m logic.src.models.core.dr_alns.example_train
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from logic.src.envs.dr_alns import DRALNSEnv
from logic.src.models.core.dr_alns import DRALNSPPOAgent, PPOTrainer


def generate_vrpp_instance(n_nodes: int = 20, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a random VRPP instance for training.

    Args:
        n_nodes: Number of customer nodes (excluding depot).
        seed: Random seed.

    Returns:
        Dictionary with instance data.
    """
    rng = np.random.RandomState(seed)

    # Generate random locations (depot at origin)
    locations = np.zeros((n_nodes + 1, 2))
    locations[1:] = rng.rand(n_nodes, 2) * 100  # 0-100 range

    # Compute distance matrix (Euclidean)
    dist_matrix = np.zeros((n_nodes + 1, n_nodes + 1))
    for i in range(n_nodes + 1):
        for j in range(n_nodes + 1):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])

    # Generate waste amounts (prizes)
    wastes = {i: rng.uniform(1.0, 10.0) for i in range(1, n_nodes + 1)}

    # Vehicle capacity
    capacity = n_nodes * 5.0  # Allow visiting most nodes

    instance = {
        "dist_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": capacity,
        "R": 1.0,  # Revenue per unit waste
        "C": 0.1,  # Cost per unit distance
        "mandatory_nodes": [],
    }

    return instance


def main():
    """Train DR-ALNS agent on VRPP instances."""
    print("=" * 60)
    print("DR-ALNS Training Example")
    print("=" * 60)

    # Configuration
    n_nodes = 20
    max_iterations = 10
    total_timesteps = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nConfiguration:")
    print("  - Problem size: {} nodes".format(n_nodes))
    print("  - Max ALNS iterations per episode: " + str(max_iterations))
    print("  - Total training timesteps: {}".format(total_timesteps))
    print("  - Device: {}".format(device))

    # Create instance generator
    def instance_generator():
        return generate_vrpp_instance(n_nodes, seed=None)

    # Create environment
    print("\nInitializing environment...")
    env = DRALNSEnv(
        max_iterations=max_iterations,
        n_destroy_ops=3,
        n_repair_ops=2,
        instance_generator=instance_generator,
    )

    # Create PPO agent
    print("Creating PPO agent...")
    agent = DRALNSPPOAgent(
        state_dim=7,
        hidden_dim=64,
        n_destroy_ops=3,
        n_repair_ops=2,
        n_severity_levels=10,
        n_temp_levels=50,
    )

    # Create trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device=device,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    history = trainer.train(
        total_timesteps=total_timesteps,
        n_steps_per_update=2048,
        log_interval=5,
        instance_generator=instance_generator,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    print("\nFinal statistics:")
    print("  - Mean episode reward: {:.2f}".format(history["mean_episode_reward"][-1]))
    print("  - Mean best profit: {:.2f}".format(history["mean_best_profit"][-1]))
    print("  - Policy loss: {:.4f}".format(history["policy_loss"][-1]))
    print("  - Value loss: {:.4f}".format(history["value_loss"][-1]))

    # Save trained agent
    save_path = "dr_alns_agent.pt"
    torch.save(agent.state_dict(), save_path)
    print(f"\nAgent saved to: {save_path}")

    # Test trained agent
    print("\n" + "=" * 60)
    print("Testing trained agent...")
    print("=" * 60 + "\n")

    test_instance = generate_vrpp_instance(n_nodes, seed=42)
    obs, info = env.reset(options={"instance": test_instance})

    total_reward = 0.0
    done = False
    step = 0

    while not done:
        state_tensor = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            actions, _, _ = agent.get_action(state_tensor, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(
            np.array(
                [
                    actions["destroy"],
                    actions["repair"],
                    actions["severity"],
                    actions["temp"],
                ]
            )
        )

        total_reward += reward
        done = terminated or truncated
        step += 1

        if step % 20 == 0 or done:
            print(
                "Step {}/{} | Current profit: {:.2f} | Best profit: {:.2f}".format(
                    step, max_iterations, info["current_profit"], info["best_profit"]
                )
            )

    print("\nTest episode completed:")
    print(f"  - Total reward: {total_reward}")
    print(f"  - Final best profit: {info['best_profit']:.2f}")
    print(f"  - Steps: {step}")


if __name__ == "__main__":
    from typing import Any, Dict, Optional

    main()
