# {py:mod}`src.models.subnets.decoders.gat.multi_head_attention_layer`

```{py:module} src.models.subnets.decoders.gat.multi_head_attention_layer
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.multi_head_attention_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiHeadAttentionLayer <src.models.subnets.decoders.gat.multi_head_attention_layer.MultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.gat.multi_head_attention_layer.MultiHeadAttentionLayer
    :summary:
    ```
````

### API

`````{py:class} MultiHeadAttentionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, **kwargs)
:canonical: src.models.subnets.decoders.gat.multi_head_attention_layer.MultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.gat.multi_head_attention_layer.MultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.multi_head_attention_layer.MultiHeadAttentionLayer.__init__
```

````{py:method} forward(q, h, mask)
:canonical: src.models.subnets.decoders.gat.multi_head_attention_layer.MultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.decoders.gat.multi_head_attention_layer.MultiHeadAttentionLayer.forward
```

````

`````
