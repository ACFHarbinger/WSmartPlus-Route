# {py:mod}`src.models.subnets.encoders.mdam.mdam_attention_layer`

```{py:module} src.models.subnets.encoders.mdam.mdam_attention_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.mdam.mdam_attention_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiHeadAttentionLayer <src.models.subnets.encoders.mdam.mdam_attention_layer.MultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.mdam.mdam_attention_layer.MultiHeadAttentionLayer
    :summary:
    ```
````

### API

`````{py:class} MultiHeadAttentionLayer(embed_dim: int, num_heads: int, feed_forward_hidden: int = 512, normalization: str = 'batch')
:canonical: src.models.subnets.encoders.mdam.mdam_attention_layer.MultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.mdam.mdam_attention_layer.MultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.mdam.mdam_attention_layer.MultiHeadAttentionLayer.__init__
```

````{py:method} forward(x: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.mdam.mdam_attention_layer.MultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.mdam.mdam_attention_layer.MultiHeadAttentionLayer.forward
```

````

`````
