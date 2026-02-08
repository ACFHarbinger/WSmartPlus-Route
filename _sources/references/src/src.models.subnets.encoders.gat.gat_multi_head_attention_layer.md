# {py:mod}`src.models.subnets.encoders.gat.gat_multi_head_attention_layer`

```{py:module} src.models.subnets.encoders.gat.gat_multi_head_attention_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_multi_head_attention_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GATMultiHeadAttentionLayer <src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer
    :summary:
    ```
````

### API

````{py:class} GATMultiHeadAttentionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, connection_type: str = 'skip', expansion_rate: int = 4, **kwargs)
:canonical: src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer

Bases: {py:obj}`logic.src.models.subnets.encoders.common.MultiHeadAttentionLayerBase`

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer.__init__
```

````
