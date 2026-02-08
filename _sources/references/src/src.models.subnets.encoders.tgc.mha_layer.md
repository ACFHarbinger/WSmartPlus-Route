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

`````{py:class} TGCMultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range)
:canonical: src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.tgc.mha_layer.TGCMultiHeadAttentionLayer.forward
```

````

`````
