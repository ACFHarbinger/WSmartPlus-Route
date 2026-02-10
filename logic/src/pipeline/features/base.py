"""
Shared utilities and base classes for pipeline features.
"""

from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf

from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.utils.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


def deep_sanitize(obj: Any) -> Any:
    """
    Recursively convert OmegaConf objects and other types to primitive types.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)

    # Use ITraversable protocol for dict-like objects
    if isinstance(obj, ITraversable):
        return {str(k): deep_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_sanitize(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def remap_legacy_keys(common_kwargs: Dict[str, Any], cfg: Config) -> None:
    """Remap config keys to legacy names for simulation compatibility."""
    # The simulation's load_model expects legacy key names
    common_kwargs["problem"] = cfg.env.name
    common_kwargs["model"] = cfg.model.name

    enc = cfg.model.encoder
    dec = cfg.model.decoder

    common_kwargs["encoder"] = enc.type
    common_kwargs["embed_dim"] = enc.embed_dim
    common_kwargs["n_encode_layers"] = enc.n_layers
    common_kwargs["n_heads"] = enc.n_heads
    common_kwargs["n_decode_layers"] = dec.n_layers
    common_kwargs["hidden_dim"] = enc.hidden_dim

    # Older/legacy fields that might not be in the new configs - providing defaults or ignoring
    common_kwargs.update(
        {
            "n_encode_sublayers": getattr(enc, "n_sublayers", None),
            "n_predict_layers": getattr(dec, "n_predictor_layers", None),
            "learn_affine": getattr(enc.normalization, "learn_affine", True) if hasattr(enc, "normalization") else True,
            "track_stats": getattr(enc.normalization, "track_stats", True) if hasattr(enc, "normalization") else True,
            "epsilon_alpha": getattr(cfg.model, "epsilon_alpha", 0.1),
            "momentum_beta": getattr(cfg.model, "momentum_beta", 0.1),
            "af_param": getattr(enc.activation, "activation_param", 0.1) if hasattr(enc, "activation") else 0.1,
            "af_threshold": getattr(enc.activation, "activation_threshold", 0.0) if hasattr(enc, "activation") else 0.0,
            "af_replacement": getattr(enc.activation, "activation_replacement", 0.0)
            if hasattr(enc, "activation")
            else 0.0,
            "af_nparams": getattr(enc.activation, "activation_num_parameters", 1) if hasattr(enc, "activation") else 1,
            "af_urange": getattr(enc.activation, "activation_uniform_range", [0, 1])
            if hasattr(enc, "activation")
            else [0, 1],
            "aggregation": getattr(cfg.model, "aggregation_node", "max"),
            "aggregation_graph": getattr(cfg.model, "aggregation_graph", "mean"),
        }
    )


def flatten_config_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested 'graph', 'reward', 'decoding', 'model', and 'policy' configs into the main dictionary.
    """
    new_dict = d.copy()

    # Flatten 'policy' first if it exists
    if "policy" in new_dict and isinstance(new_dict["policy"], ITraversable):
        policy_cfg = new_dict.pop("policy")
        new_dict.update(policy_cfg)

    # Flatten other standard sub-configs
    for key in ["graph", "reward", "decoding", "model"]:
        if key in new_dict and isinstance(new_dict[key], ITraversable):
            nested = new_dict.pop(key)
            new_dict.update(nested)

    return new_dict
