"""
Configuration dataclasses for WSmart-Route.

Attributes:
    Config: Root configuration.

Example:
    >>> from logic.src.configs import Config
    >>> config = Config()
    >>> print(config)
    Config(env=EnvConfig(max_nodes=50, n_customers=20, customer_types={'A': 0.25, 'B': 0.5, 'C': 0.25}, n_depots=1, demand_distribution='normal', demand_normal_mean=10, demand_normal_std=1, min_demand=1, max_demand=25, capacity_distribution='normal', capacity_normal_mean=300, capacity_normal_std=50, min_capacity=200, max_capacity=400, instance_generator='random', max_travel_time=24, opening_time=0, closing_time=24, min_waste_nodes_ratio=0.4, max_waste_nodes_ratio=0.6, service_time=1, n_time_bins=1, max_time_bin_length=1, generate_empty_graphs=False, n_edges_factor=3, shuffle_seed=42), model=ModelConfig(type='Transformer', encoder_hidden_dim=64, decoder_hidden_dim=64, n_encoder_layers=3, n_decoder_layers=3, n_heads=4, dropout_rate=0.1, normalization='layer', activation='gelu', feed_forward_expansion=4, use_skip_connections=True, tanh_clipping=10.0, predictor_layers=None), hpo=HPOConfig(type='optuna', n_trials=100, n_jobs=-1, show_progress=True, pruning=True, pruning_n_trials=10, pruning_n_startup_trials=5, pruning_n_intermediate_trials=5, pruner='halving', sampler='tpe', storage=None), hpo_sim=SimHPOConfig(n_trials=1000, n_jobs=-1, show_progress=True, pruning=True, pruning_n_trials=10, pruning_n_startup_trials=5, pruning_n_intermediate_trials=5, pruner='halving', sampler='tpe', storage=None), rl=RLConfig(gamma=1.0, learning_rate=0.001, buffer_size=10000, min_buffer_size=100, batch_size=32, n_epochs=10, n_warmup_samples=0, update_rate=0.01, target_update_rate=0.005, discount_factor=0.9, reward_scale=1.0, n_greedy_episodes=20, n_exploit_episodes=5, eval_every_n_steps=100, eval_n_episodes=20, eval_baseline_n_episodes=20, log_interval=10, train_batch_size=32, n_train_batches=50, n_eval_episodes=10, n_test_episodes=100, target_network_update_freq=500, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05, train_value_net=True, value_net_learning_rate=0.001, value_net_hidden_dim=64, value_net_layers=2, value_net_dropout=0.1, gradient_clipping=1.0, n_steps=1, n_target_update_steps=1, train_freq=4, gradient_accumulation_steps=1, use_gae=False, gae_lambda=0.95, clip_range=0.2, clip_range_vf=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=1.0, rollout_buffer_size=1024, learning_starts=0, train_freq_steps=1, train_batch_size_steps=32, max_grad_norm=1.0), meta_rl=MetaRLConfig(total_updates=10000, meta_batch_size=32, meta_learning_rate=0.001, inner_learning_rate=0.01, inner_updates=1, embedding_size=32, embedding_lr=0.001, num_heads=4, num_layers=2, use_skip_connections=True, activation='gelu', normalization='layer', dropout_rate=0.1, tanh_clipping=10.0, predictor_layers=None, rollout_buffer_size=1024, min_rollout_buffer_size=100, train_freq=4, gradient_accumulation_steps=1, clip_range=0.2, clip_range_vf=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=1.0, train_freq_steps=1, train_batch_size_steps=32), train=TrainConfig(max_epochs=100, batch_size=32, learning_rate=0.001, dropout_rate=0.1, clip_grad_norm=None, weight_decay=0.0, label_smoothing=0.0, warmup_epochs=10, reduce_lr_on_plateau=False, patience=5, min_lr=1e-06, patience_epochs=5, eval_freq_epochs=10, eval_batch_size=None, gradient_accumulation_steps=1, grad_clip=1.0, label_smoothing=0.0, gradient_clipping=1.0, gradient_clipping_norm=None, target_temperature=1.0, target_temperature_decay_rate=0.999997, temperature_decay_steps=10000, learning_rate_decay=None, learning_rate_decay_rate=0.9999, patience=5, min_learning_rate=1e-5), optim=OptimConfig(type='adamw', lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0, amsgrad=False, foreach=False, grad_clip_norm=None, gradient_clipping=None), data=DataConfig(data_dir="", train_size=100, val_size=20, test_size=20, n_train_workers=4, n_val_workers=2, n_test_workers=2, shuffle=True, pin_memory=False, persistent_workers=False, batch_size=32, val_batch_size=32, test_batch_size=32), eval=EvalConfig(eval_size=20, n_workers=2, batch_size=32, shuffle=True, pin_memory=False, persistent_workers=False), sim=SimConfig(n_agents=1, n_simulations=1000, n_parallel_sims=10, visualize=True, batch_size=32, val_batch_size=32, test_batch_size=32, val_size=20, test_size=20, n_train_workers=0, n_test_workers=0), hpo=HPOConfig(n_trials=100, n_jobs=-1, show_progress=True, pruning=True, pruning_n_trials=10, pruning_n_startup_trials=5, pruning_n_intermediate_trials=5, pruner='halving', sampler='tpe', storage=None), tracking=TrackingConfig(name="", project_name=None, use_mlflow=False, use_tensorboard=False, use_wandb=True), mandatory_selection=MandatorySelectionConfig(nodes_ratio=0.2, edges_ratio=0.0, strategy='random'), route_improvement=RouteImprovingConfig(n_iterations=10, swap_cost_weight=0.0001, insertion_cost_weight=0.0001, max_neighbors=1000), load_dataset=None, seed=42, device='cuda', experiment_name=None, task='train', output_dir='assets/model_weights', log_dir='assets/logs', run_name=None, verbose=True, start=0, p={}, callbacks={}),
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .envs import EnvConfig, GraphConfig, ObjectiveConfig
from .models import DecoderConfig, DecodingConfig, EncoderConfig, ModelConfig, OptimConfig
from .policies.other import MandatorySelectionConfig, RouteImprovingConfig
from .rl import RLConfig
from .tasks import DataConfig, EvalConfig, HPOConfig, MetaRLConfig, SimConfig, SimHPOConfig, TrainConfig
from .tracking import TrackingConfig


@dataclass
class Config:
    """Root configuration.

    Attributes:
        env: Environment configuration.
        model: Model configuration.
        train: Training configuration.
        optim: Optimizer configuration.
        rl: RL algorithm configuration.
        meta_rl: Meta-RL configuration.
        hpo: HPO configuration.
        eval: Evaluation configuration.
        sim: Simulation configuration.
        data: Data generation configuration.
        tracking: Tracking backend configuration (WSTracker + optional MLflow).
        mandatory_selection: Mandatory nodes selection strategy configuration.
        route_improvement: Route refinement configuration.
        load_dataset: Optional path to a dataset file to load.
        seed: Random seed.
        device: Device to use ('cpu', 'cuda').
        experiment_name: Optional name for the experiment.
        task: Task to perform ('train', 'eval', 'test_sim', 'gen_data').
        wandb_mode: Weights & Biases mode ('online', 'offline', 'disabled').
        no_tensorboard: If True, disable TensorBoard logging.
        no_progress_bar: If True, disable the progress bar.
        output_dir: Directory to save model outputs and artifacts.
        log_dir: Directory to save logs.
        run_name: Specific name for the run (separate from experiment_name).
        verbose: If True, enable verbose logging.
        start: Starting index or offset (e.g., for resuming or dataset slicing).
        p: Dictionary for arbitrary additional parameters.
        callbacks: Dictionary of callback configurations.
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    meta_rl: MetaRLConfig = field(default_factory=MetaRLConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    hpo_sim: SimHPOConfig = field(default_factory=SimHPOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mandatory_selection: MandatorySelectionConfig = field(default_factory=MandatorySelectionConfig)
    route_improvement: RouteImprovingConfig = field(default_factory=RouteImprovingConfig)
    load_dataset: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    task: str = "train"
    output_dir: str = "assets/model_weights"
    run_name: Optional[str] = None
    start: int = 0
    p: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "OptimConfig",
    "RLConfig",
    "MetaRLConfig",
    "HPOConfig",
    "EvalConfig",
    "SimConfig",
    "SimHPOConfig",
    "DataConfig",
    "TrackingConfig",
    "MandatorySelectionConfig",
    "RouteImprovingConfig",
    "Config",
    "EncoderConfig",
    "DecoderConfig",
    "DecodingConfig",
    "GraphConfig",
    "ObjectiveConfig",
]
