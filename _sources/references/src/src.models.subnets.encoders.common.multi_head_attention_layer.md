# {py:mod}`src.models.subnets.encoders.common.multi_head_attention_layer`

```{py:module} src.models.subnets.encoders.common.multi_head_attention_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.common.multi_head_attention_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiHeadAttentionLayerBase <src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase
    :summary:
    ```
````

### API

`````{py:class} MultiHeadAttentionLayerBase(n_heads: int, embed_dim: int, feed_forward_hidden: int, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, connection_type: str = 'skip', expansion_rate: int = 4, **kwargs)
:canonical: src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase.__init__
```

````{py:method} _create_feed_forward(embed_dim: int, feed_forward_hidden: int, activation_config: logic.src.configs.models.activation_function.ActivationConfig, connection_type: str, expansion_rate: int) -> torch.nn.Module
:canonical: src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase._create_feed_forward

```{autodoc2-docstring} src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase._create_feed_forward
```

````

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase.forward

```{autodoc2-docstring} src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase.forward
```

````

````{py:method} _apply_norm(h: torch.Tensor, norm_layer: torch.nn.Module) -> torch.Tensor
:canonical: src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase._apply_norm

```{autodoc2-docstring} src.models.subnets.encoders.common.multi_head_attention_layer.MultiHeadAttentionLayerBase._apply_norm
```

````

`````
