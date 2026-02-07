"""
Shared utilities and base classes for pipeline features.
"""

from typing import Any, Dict

from logic.src.configs import Config
from logic.src.utils.logging.pylogger import get_pylogger
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = get_pylogger(__name__)


def deep_sanitize(obj: Any) -> Any:
    """
    Recursively convert OmegaConf objects and other types to primitive types.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)

    if isinstance(obj, dict):
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
    common_kwargs["encoder"] = cfg.model.encoder_type
    common_kwargs["embed_dim"] = cfg.model.embed_dim
    common_kwargs["n_encode_layers"] = cfg.model.n_encode_layers
    common_kwargs["n_encode_sublayers"] = cfg.model.n_encoder_sublayers
    common_kwargs["n_decode_layers"] = cfg.model.n_decode_layers
    common_kwargs["n_heads"] = cfg.model.n_heads
    common_kwargs["n_predict_layers"] = cfg.model.n_predictor_layers
    common_kwargs["learn_affine"] = cfg.model.learn_affine
    common_kwargs["track_stats"] = cfg.model.track_stats
    common_kwargs["epsilon_alpha"] = cfg.model.epsilon_alpha
    common_kwargs["momentum_beta"] = cfg.model.momentum_beta
    common_kwargs["af_param"] = cfg.model.activation_param
    common_kwargs["af_threshold"] = cfg.model.activation_threshold
    common_kwargs["af_replacement"] = cfg.model.activation_replacement
    common_kwargs["af_nparams"] = cfg.model.activation_num_parameters
    common_kwargs["af_urange"] = cfg.model.activation_uniform_range
    common_kwargs["aggregation"] = cfg.model.aggregation_node
    common_kwargs["aggregation_graph"] = cfg.model.aggregation_graph
    common_kwargs["hidden_dim"] = cfg.model.hidden_dim
