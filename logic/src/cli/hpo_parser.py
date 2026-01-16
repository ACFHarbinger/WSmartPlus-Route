"""
Hyper-Parameter Optimization argument parser.
"""

from logic.src.cli.train_parser import add_train_args


def add_hp_optim_args(parser):
    """
    Adds arguments for Hyper-Parameter Optimization (inherits from train_args).

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added HPO arguments.
    """
    parser = add_train_args(parser)

    # HPO specific args
    parser.add_argument(
        "--hop_method",
        type=str,
        default="dehbo",
        choices=["dea", "bo", "hbo", "rs", "gs", "dehbo"],
        help="Method to use for hyperparameter optimization",
    )
    parser.add_argument(
        "--hop_range",
        type=float,
        nargs="+",
        default=[0.0, 2.0],
        help="Maximum and minimum values for hyperparameter optimization",
    )
    parser.add_argument(
        "--hop_epochs",
        type=int,
        default=7,
        help="The number of epochs to optimize hyperparameters",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_loss",
        choices=[
            "loss",
            "val_loss",
            "mean_reward",
            "mae",
            "mse",
            "rmse",
            "episode_reward_mean",
            "kg/km",
            "overflows",
            "both",
        ],
        help="Metric to optimize",
    )

    # ===== Bayesian Optimization (BO) =====
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for Optuna optimization",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for Optuna optimization (in seconds)",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Number of trials to run before pruning starts",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=3,
        help="Number of epochs to wait before pruning can happen in each trial",
    )
    parser.add_argument(
        "--interval_steps",
        type=int,
        default=1,
        help="Pruning is evaluated every this many epochs",
    )

    # ===== Distributed Evolutionary Algorithm (DEA) =====
    parser.add_argument(
        "--eta",
        type=float,
        default=10.0,
        help="Controls the spread of the genetic mutations (higher = slower changes)",
    )
    parser.add_argument(
        "--indpb",
        type=float,
        default=0.2,
        help="Probability of mutating each gene of an individual in the population",
    )
    parser.add_argument(
        "--tournsize",
        type=int,
        default=3,
        help="Number of individuals to fight to be selected for reproduction",
    )
    parser.add_argument(
        "--cxpb",
        type=float,
        default=0.7,
        help="Probability of crossover between two parents (higher = faster convergence + less diversity)",
    )
    parser.add_argument(
        "--mutpb",
        type=float,
        default=0.2,
        help="Probability of an individual being mutated after crossover",
    )
    parser.add_argument(
        "--n_pop",
        type=int,
        default=20,
        help="Starting population for evolutionary algorithms",
    )
    parser.add_argument("--n_gen", type=int, default=10, help="Number of generations to evolve")

    # ===== Differential Evolutionary Hyperband Optimization (DEHBO) =====
    parser.add_argument("--fevals", type=int, default=100, help="Number of function evaluations")

    # Ray Tune framework hyperparameters
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=1,
        help="Number of CPU cores to use for hyperparameter optimization (0 uses all available cores)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity level for Hyperband and Random Search (0-3)",
    )
    parser.add_argument(
        "--train_best",
        action="store_true",
        default=True,
        help="Train final model with best hyperparameters",
    )
    parser.add_argument(
        "--local_mode",
        action="store_true",
        help="Run ray in local mode (for debugging)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of times to sample from the hyperparameter space",
    )

    # ===== Hyperband Optimization (HBO) - Ray Tune =====
    parser.add_argument(
        "--max_tres",
        type=int,
        default=14,
        help="Maximum resources (timesteps) per trial",
    )
    parser.add_argument("--reduction_factor", type=int, default=3, help="Reduction factor for Hyperband")

    # ===== Random Search (RS) - Ray Tune =====
    parser.add_argument(
        "--max_failures",
        type=int,
        default=3,
        help="Maximum trial failures before stopping",
    )

    # ===== Grid Search (GS) - Ray Tune =====
    parser.add_argument(
        "--grid",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Hyperparameter values to try in grid search",
    )
    parser.add_argument(
        "--max_conc",
        type=int,
        default=4,
        help="Maximum number of concurrent trials for Ray Tune",
    )
    return parser
