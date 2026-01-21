"""
New PyTorch Lightning training entry point using Hydra.
"""
import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import seed_everything

from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.models.policies import (
    AttentionModelPolicy,
    DeepDecoderPolicy,
    PointerNetworkPolicy,
    TemporalAMPolicy,
)
from logic.src.pipeline.rl import REINFORCE
from logic.src.pipeline.trainer import WSTrainer

# Register configuration
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    """
    Main training function using Hydra and PyTorch Lightning.
    """
    # 1. Set seed
    seed_everything(cfg.seed)

    # 2. Initialize Environment
    # cfg.env is EnvConfig, passed as kwargs
    env = get_env(cfg.env.name, **vars(cfg.env))

    # 3. Initialize Policy
    policy_map = {
        "am": AttentionModelPolicy,
        "deep_decoder": DeepDecoderPolicy,
        "temporal": TemporalAMPolicy,
        "pointer": PointerNetworkPolicy,
    }

    if cfg.model.name not in policy_map:
        raise ValueError(f"Unknown model name: {cfg.model.name}. Available: {list(policy_map.keys())}")

    policy_cls = policy_map[cfg.model.name]

    # Common policy kwargs
    policy_kwargs = {
        "env_name": cfg.env.name,
        "embed_dim": cfg.model.embed_dim,
        "hidden_dim": cfg.model.hidden_dim,
        "n_encode_layers": cfg.model.num_encoder_layers,
        "n_heads": cfg.model.num_heads,
        "normalization": "batch",
    }

    # Model-specific kwargs
    if cfg.model.name == "deep_decoder":
        policy_kwargs["n_decode_layers"] = cfg.model.num_decoder_layers

    policy = policy_cls(**policy_kwargs)

    # 4. Initialize RL Module
    common_kwargs = {
        "env": env,
        "policy": policy,
        "optimizer": cfg.optim.optimizer,
        "optimizer_kwargs": {"lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
        "lr_scheduler": cfg.optim.lr_scheduler,
        "lr_scheduler_kwargs": cfg.optim.lr_scheduler_kwargs,
        "train_data_size": cfg.train.train_data_size,
        "val_data_size": cfg.train.val_data_size,
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "entropy_weight": cfg.rl.entropy_weight,
        "max_grad_norm": cfg.rl.max_grad_norm,
    }

    if cfg.rl.algorithm == "ppo":
        from logic.src.models.policies.critic import CriticNetwork
        from logic.src.pipeline.rl import PPO

        critic = CriticNetwork(
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model = PPO(
            critic=critic,
            ppo_epochs=cfg.rl.ppo_epochs,
            eps_clip=cfg.rl.eps_clip,
            value_loss_weight=cfg.rl.value_loss_weight,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "sapo":
        from logic.src.models.policies.critic import CriticNetwork
        from logic.src.pipeline.rl import SAPO

        critic = CriticNetwork(
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model = SAPO(
            critic=critic,
            tau_pos=cfg.rl.sapo_tau_pos,
            tau_neg=cfg.rl.sapo_tau_neg,
            ppo_epochs=cfg.rl.ppo_epochs,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "gspo":
        from logic.src.models.policies.critic import CriticNetwork
        from logic.src.pipeline.rl import GSPO

        critic = CriticNetwork(
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model = GSPO(
            critic=critic,
            ppo_epochs=cfg.rl.ppo_epochs,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "dr_grpo":
        # DRGRPO often doesn't use a critic, but we'll allow it if needed
        # For now, let's just pass dummy critic or None if the class can handle it
        # PPO base requires a critic, so we'll provide one
        from logic.src.models.policies.critic import CriticNetwork
        from logic.src.pipeline.rl import DRGRPO

        critic = CriticNetwork(
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model = DRGRPO(
            critic=critic,
            ppo_epochs=cfg.rl.ppo_epochs,
            **common_kwargs,
        )
    else:
        model = REINFORCE(
            baseline=cfg.rl.baseline,
            **common_kwargs,
        )

    # 5. Initialize Trainer
    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=1 if cfg.device == "cuda" else "auto",
        gradient_clip_val=None if cfg.rl.algorithm == "ppo" else cfg.rl.max_grad_norm,
        logger=False,
    )

    # 6. Train
    trainer.fit(model)


if __name__ == "__main__":
    main()
