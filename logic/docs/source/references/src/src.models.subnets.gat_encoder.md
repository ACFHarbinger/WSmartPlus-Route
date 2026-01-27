# {py:mod}`src.models.subnets.gat_encoder`

```{py:module} src.models.subnets.gat_encoder
```

```{autodoc2-docstring} src.models.subnets.gat_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FeedForwardSubLayer <src.models.subnets.gat_encoder.FeedForwardSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.gat_encoder.FeedForwardSubLayer
    :summary:
    ```
* - {py:obj}`MultiHeadAttentionLayer <src.models.subnets.gat_encoder.MultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.gat_encoder.MultiHeadAttentionLayer
    :summary:
    ```
* - {py:obj}`GraphAttentionEncoder <src.models.subnets.gat_encoder.GraphAttentionEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.gat_encoder.GraphAttentionEncoder
    :summary:
    ```
````

### API

`````{py:class} FeedForwardSubLayer(embed_dim: int, feed_forward_hidden: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, dist_range: typing.List[float], bias: bool = True)
:canonical: src.models.subnets.gat_encoder.FeedForwardSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gat_encoder.FeedForwardSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gat_encoder.FeedForwardSubLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.gat_encoder.FeedForwardSubLayer.forward

```{autodoc2-docstring} src.models.subnets.gat_encoder.FeedForwardSubLayer.forward
```

````

`````

`````{py:class} MultiHeadAttentionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, normalization: str, epsilon_alpha: float, learn_affine: bool, track_stats: bool, mbeta: float, lr_k: float, n_groups: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, uniform_range: typing.List[float], connection_type: str = 'skip', expansion_rate: int = 4)
:canonical: src.models.subnets.gat_encoder.MultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gat_encoder.MultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gat_encoder.MultiHeadAttentionLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.gat_encoder.MultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.gat_encoder.MultiHeadAttentionLayer.forward
```

````

`````

`````{py:class} GraphAttentionEncoder(n_heads: int, embed_dim: int, n_layers: int, n_sublayers: typing.Optional[int] = None, feed_forward_hidden: int = 512, normalization: str = 'batch', epsilon_alpha: float = 1e-05, learn_affine: bool = True, track_stats: bool = False, momentum_beta: float = 0.1, locresp_k: float = 1.0, n_groups: int = 3, activation: str = 'gelu', af_param: float = 1.0, threshold: float = 6.0, replacement_value: float = 6.0, n_params: int = 3, uniform_range: typing.List[float] = None, dropout_rate: float = 0.1, agg: typing.Any = None, connection_type: str = 'skip', expansion_rate: int = 4, **kwargs)
:canonical: src.models.subnets.gat_encoder.GraphAttentionEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gat_encoder.GraphAttentionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gat_encoder.GraphAttentionEncoder.__init__
```

````{py:method} forward(x: torch.Tensor, edges: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.gat_encoder.GraphAttentionEncoder.forward

```{autodoc2-docstring} src.models.subnets.gat_encoder.GraphAttentionEncoder.forward
```

````

`````
