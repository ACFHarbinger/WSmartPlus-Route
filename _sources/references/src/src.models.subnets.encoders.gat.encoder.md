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

`````{py:class} GraphAttentionEncoder(n_heads: int, embed_dim: int, n_layers: int, n_sublayers: typing.Optional[int] = None, feed_forward_hidden: int = 512, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, dropout_rate: float = 0.1, agg: typing.Any = None, connection_type: str = 'skip', expansion_rate: int = 4, **kwargs)
:canonical: src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder

Bases: {py:obj}`logic.src.models.subnets.encoders.common.TransformerEncoderBase`

```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder.__init__
```

````{py:method} _create_layer(layer_idx: int) -> torch.nn.Module
:canonical: src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder._create_layer

```{autodoc2-docstring} src.models.subnets.encoders.gat.encoder.GraphAttentionEncoder._create_layer
```

````

`````
