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

`````{py:class} GATMultiHeadAttentionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, normalization: str, epsilon_alpha: float, learn_affine: bool, track_stats: bool, mbeta: float, lr_k: float, n_groups: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, uniform_range: typing.List[float], connection_type: str = 'skip', expansion_rate: int = 4)
:canonical: src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_multi_head_attention_layer.GATMultiHeadAttentionLayer.forward
```

````

`````
