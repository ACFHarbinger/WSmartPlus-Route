# {py:mod}`src.models.subnets.encoders.ggac.encoder`

```{py:module} src.models.subnets.encoders.ggac.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GatedGraphAttConvEncoder <src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder
    :summary:
    ```
````

### API

`````{py:class} GatedGraphAttConvEncoder(n_heads: int, embed_dim: int, n_layers: int, n_sublayers: typing.Optional[int] = None, feed_forward_hidden: int = 512, normalization: str = 'batch', epsilon_alpha: float = 1e-05, learn_affine: bool = True, track_stats: bool = False, momentum_beta: float = 0.1, locresp_k: float = 1.0, n_groups: int = 3, activation: str = 'gelu', af_param: float = 1.0, threshold: float = 6.0, replacement_value: float = 6.0, n_params: int = 3, uniform_range: typing.Optional[list] = None, dropout_rate: float = 0.1, agg: str = 'sum', **kwargs)
:canonical: src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder

Bases: {py:obj}`logic.src.models.subnets.encoders.common.TransformerEncoderBase`

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder.__init__
```

````{py:method} _create_layer(layer_idx: int) -> torch.nn.Module
:canonical: src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder._create_layer

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder._create_layer
```

````

````{py:method} forward(x: torch.Tensor, edges: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, dist: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder.forward
```

````

`````
