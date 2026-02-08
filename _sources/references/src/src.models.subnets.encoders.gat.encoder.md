# {py:mod}`src.models.subnets.encoders.gat.encoder`

```{py:module} src.models.subnets.encoders.gat.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphAttentionEncoder <src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder
    :summary:
    ```
````

### API

`````{py:class} GraphAttentionEncoder(n_heads: int, embed_dim: int, n_layers: int, n_sublayers: typing.Optional[int] = None, feed_forward_hidden: int = 512, normalization: str = 'batch', epsilon_alpha: float = 1e-05, learn_affine: bool = True, track_stats: bool = False, momentum_beta: float = 0.1, locresp_k: float = 1.0, n_groups: int = 3, activation: str = 'gelu', af_param: float = 1.0, threshold: float = 6.0, replacement_value: float = 6.0, n_params: int = 3, uniform_range: typing.List[float] = None, dropout_rate: float = 0.1, agg: typing.Any = None, connection_type: str = 'skip', expansion_rate: int = 4, **kwargs)
:canonical: src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder.__init__
```

````{py:method} forward(x: torch.Tensor, edges: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder.forward
```

````

`````
