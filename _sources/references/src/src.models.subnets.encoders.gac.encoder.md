# {py:mod}`src.models.subnets.encoders.gac.encoder`

```{py:module} src.models.subnets.encoders.gac.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphAttConvEncoder <src.models.subnets.encoders.gac.encoder.GraphAttConvEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gac.encoder.GraphAttConvEncoder
    :summary:
    ```
````

### API

`````{py:class} GraphAttConvEncoder(n_heads: int, embed_dim: int, n_layers: int, n_sublayers: typing.Optional[int] = None, feed_forward_hidden: int = 512, normalization: str = 'batch', epsilon_alpha: float = 1e-05, learn_affine: bool = True, track_stats: bool = False, momentum_beta: float = 0.1, locresp_k: float = 1.0, n_groups: int = 3, activation: str = 'gelu', af_param: float = 1.0, threshold: float = 6.0, replacement_value: float = 6.0, n_params: int = 3, uniform_range: typing.Optional[list] = None, dropout_rate: float = 0.1, aggregate: str = 'sum', **kwargs)
:canonical: src.models.subnets.encoders.gac.encoder.GraphAttConvEncoder

Bases: {py:obj}`logic.src.models.subnets.encoders.common.TransformerEncoderBase`

```{autodoc2-docstring} src.models.subnets.encoders.gac.encoder.GraphAttConvEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.encoder.GraphAttConvEncoder.__init__
```

````{py:method} _create_layer(layer_idx: int) -> torch.nn.Module
:canonical: src.models.subnets.encoders.gac.encoder.GraphAttConvEncoder._create_layer

```{autodoc2-docstring} src.models.subnets.encoders.gac.encoder.GraphAttConvEncoder._create_layer
```

````

`````
