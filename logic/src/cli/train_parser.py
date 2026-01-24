"""
Argument parsing for training and optimization commands.
"""

from logic.src.cli.base_parser import LowercaseAction


def add_train_args(parser):
    """Adds arguments for the 'train' command."""
    parser.add_argument("--problem", type=str, default="vrpp", action=LowercaseAction)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_size", type=int, default=128000)
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--model", type=str, default="am", action=LowercaseAction)
    parser.add_argument("--encoder", type=str, default="gat", action=LowercaseAction)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_encode_layers", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam", action=LowercaseAction)
    parser.add_argument("--lr_model", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default=None, action=LowercaseAction)
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--baseline", type=str, default=None, action=LowercaseAction)
    parser.add_argument("--bl_warmup_epochs", type=int, default=0)
    parser.add_argument("--area", type=str, default="riomaior", action=LowercaseAction)
    parser.add_argument("--edge_threshold", type=lambda x: int(x) if x.isdigit() else float(x), default=None)
    parser.add_argument("--activation", type=str, default="relu", action=LowercaseAction)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--enable_scaler", action="store_true")
    parser.add_argument("--waste_type", type=str, action=LowercaseAction)
    parser.add_argument("--af_urange", nargs=2, type=float, default=[0.125, 0.3333333333333333])
    parser.add_argument("--lrs_milestones", nargs="+", type=int)


def add_hp_optim_args(parser):
    """Adds arguments for the 'hp_optim' command."""
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_size", type=int, default=128000)
    parser.add_argument("--hop_method", type=str, default="bo", action=LowercaseAction)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--metric", type=str, default="val_loss", action=LowercaseAction)
    parser.add_argument("--grid", nargs="+", type=float)
    parser.add_argument("--n_pop", type=int, default=20)
    parser.add_argument("--n_gen", type=int, default=10)
    parser.add_argument("--mutpb", type=float, default=0.2)
    parser.add_argument("--lrs_milestones", nargs="+", type=int)


def add_mrl_train_args(parser):
    """Adds arguments for the 'mrl_train' command."""
    parser.add_argument("--problem", type=str, default="vrpp", action=LowercaseAction)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_size", type=int, default=128000)
    parser.add_argument("--mrl_method", type=str, default="cb", action=LowercaseAction)
    parser.add_argument("--mrl_history", type=int, default=10)
    parser.add_argument("--cb_exploration_method", type=str, default="epsilon_greedy", action=LowercaseAction)
    parser.add_argument("--cb_num_configs", type=int, default=5)
    parser.add_argument("--cb_epsilon_decay", type=float, default=0.99)


def validate_train_args(opts):
    """Validates training arguments."""
    if opts.get("epoch_size") and opts.get("batch_size"):
        assert opts["epoch_size"] % opts["batch_size"] == 0, "Epoch size must be integer multiple of batch size"

    if opts.get("baseline") == "rollout" and opts.get("bl_warmup_epochs") == 0:
        opts["bl_warmup_epochs"] = 1

    if opts.get("area"):
        opts["area"] = opts["area"].replace("-", "").lower()

    if opts.get("waste_type"):
        opts["waste_type"] = opts["waste_type"].replace("-", "").lower()

    return opts


def validate_hp_optim_args(opts):
    """Validates HPO arguments."""
    return opts


def validate_mrl_train_args(opts):
    """Validates MRL training arguments."""
    return opts
