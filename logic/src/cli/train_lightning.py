"""
New PyTorch Lightning training entry point using Hydra.
"""
import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import seed_everything

from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.models.policies import AttentionModelPolicy
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
    # Currently hardcoded to AttentionModelPolicy for migration
    policy = AttentionModelPolicy(
        env_name=cfg.env.name,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_encode_layers=cfg.model.num_encoder_layers,
        n_heads=cfg.model.num_heads,
        normalization="batch",  # Defaulting for migration
    )

    # 4. Initialize RL Module
    model = REINFORCE(
        env=env,
        policy=policy,
        baseline=cfg.rl.baseline,
        optimizer=cfg.optim.optimizer,
        optimizer_kwargs={"lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
        lr_scheduler=cfg.optim.lr_scheduler,
        lr_scheduler_kwargs=cfg.optim.lr_scheduler_kwargs,
        train_data_size=cfg.train.train_data_size,
        val_data_size=cfg.train.val_data_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        entropy_weight=cfg.rl.entropy_weight,
        max_grad_norm=cfg.rl.max_grad_norm,
    )

    # 5. Initialize Trainer
    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=1 if cfg.device == "cuda" else "auto",
        logger=False,
    )

    # 6. Train
    trainer.fit(model)


if __name__ == "__main__":
    main()
