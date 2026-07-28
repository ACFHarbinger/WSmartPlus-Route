# {py:mod}`src.models.subnets.encoders.tgc.mha_layer`

```{py:module} src.models.subnets.encoders.tgc.mha_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TGCMultiHeadAttentionLayer <src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer
    :summary:
    ```
````

### API

`````{py:class} TGCMultiHeadAttentionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, normalization: str, epsilon_alpha: float, learn_affine: bool, track_stats: bool, mbeta: float, lr_k: float, n_groups: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, uniform_range: typing.List[float])
:canonical: src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer.forward
```

````

`````
