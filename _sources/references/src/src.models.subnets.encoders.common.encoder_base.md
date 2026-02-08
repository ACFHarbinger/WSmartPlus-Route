# {py:mod}`src.models.subnets.encoders.common.encoder_base`

```{py:module} src.models.subnets.encoders.common.encoder_base
```

```{autodoc2-docstring} src.models.subnets.encoders.common.encoder_base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TransformerEncoderBase <src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase
    :summary:
    ```
````

### API

`````{py:class} TransformerEncoderBase(n_heads: int, embed_dim: int, n_layers: int, feed_forward_hidden: int = 512, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, dropout_rate: float = 0.1, connection_type: str = 'skip', expansion_rate: int = 4, **kwargs: typing.Any)
:canonical: src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase.__init__
```

````{py:method} _create_layer(layer_idx: int) -> torch.nn.Module
:canonical: src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase._create_layer
:abstractmethod:

```{autodoc2-docstring} src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase._create_layer
```

````

````{py:method} forward(x: torch.Tensor, edges: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase.forward

```{autodoc2-docstring} src.models.subnets.encoders.common.encoder_base.TransformerEncoderBase.forward
```

````

`````
