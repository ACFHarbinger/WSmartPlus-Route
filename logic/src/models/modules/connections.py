
from .skip_connection import SkipConnection
from .hyper_connection import StaticHyperConnection, DynamicHyperConnection

def get_connection_module(conn_type, module, embed_dim, expansion_rate=4):
    if conn_type == 'residual':
        return SkipConnection(module)
    elif conn_type == 'static_hyper':
        return StaticHyperConnection(module, embed_dim, expansion_rate)
    elif conn_type == 'dynamic_hyper':
        return DynamicHyperConnection(module, embed_dim, expansion_rate)
    else:
        raise ValueError(f"Unknown connection type: {conn_type}")
