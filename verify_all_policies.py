from logic.src.cli.train_lightning import create_model
from logic.src.configs import Config, EnvConfig, ModelConfig, RLConfig, TrainConfig


def test_policy(policy_name, problem="vrpp"):
    print(f"Testing policy: {policy_name} on {problem}...")
    cfg = Config()
    cfg.env = EnvConfig(name=problem, graph_size=20)
    cfg.model = ModelConfig(name=policy_name)
    cfg.rl = RLConfig(algorithm="reinforce")
    cfg.train = TrainConfig(train_data_size=64, val_data_size=32, batch_size=16)

    try:
        model = create_model(cfg)
        print(f"Successfully created {policy_name} model.")

        # Dummy batch
        td = model.env.reset(batch_size=[2])
        out = model.policy(td, model.env)
        print(f"Successfully ran forward pass for {policy_name}. Reward shape: {out['reward'].shape}")

    except Exception as e:
        print(f"Error testing {policy_name}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    policies = ["am", "alns", "hgs", "hybrid"]
    for p in policies:
        test_policy(p)
