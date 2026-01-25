"""
Training related argument parsers.
"""


def add_train_args(parser):
    """Adds training arguments to the parser."""
    parser.add_argument("--model", type=str, default="am", help="Model type")
    parser.add_argument("--problem", type=str, default="vrpp", help="Problem type")
    parser.add_argument("--graph_size", type=int, default=20, help="Graph size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    return parser


def add_mrl_train_args(parser):
    """Adds Meta-RL training arguments."""
    parser = add_train_args(parser)
    parser.add_argument("--meta_lr", type=float, default=1e-3, help="Meta learning rate")
    return parser


def add_hp_optim_args(parser):
    """Adds hyperparameter optimization arguments."""
    parser = add_train_args(parser)
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    return parser


def validate_train_args(args):
    """Validates training arguments."""
    # Add minimal validation logic if necessary
    return args
