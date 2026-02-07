"""
Training engine for WSmart-Route.
"""

import torch
from logic.src.configs import Config
from logic.src.pipeline.callbacks import SpeedMonitor
from logic.src.pipeline.features.train.model_factory import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger


def run_training(cfg: Config) -> float:
    """Run single model training."""
    seed_everything(cfg.seed)

    # Enable Tensor Core acceleration for Ampere+ GPUs
    if torch.cuda.is_available() and cfg.train.precision in ["16-mixed", "bf16-mixed"]:
        torch.set_float32_matmul_precision("medium")

    model = create_model(cfg)

    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        gradient_clip_val=(float(cfg.rl.max_grad_norm) if cfg.rl.algorithm != "ppo" else 0.0),
        logger=CSVLogger(cfg.train.logs_dir or "logs", name=""),
        callbacks=[SpeedMonitor(epoch_time=True)],
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_step,
        model_weights_path=cfg.train.model_weights_path,
        logs_dir=cfg.train.logs_dir,
        reload_dataloaders_every_n_epochs=cfg.train.reload_dataloaders_every_n_epochs,
    )

    trainer.fit(model)

    # Save final weights if path is provided
    if cfg.train.final_model_path:
        model.save_weights(cfg.train.final_model_path)

    return trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()
