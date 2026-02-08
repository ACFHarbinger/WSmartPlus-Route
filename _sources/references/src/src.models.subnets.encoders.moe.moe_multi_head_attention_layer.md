# {py:mod}`src.models.subnets.encoders.moe.moe_multi_head_attention_layer`

```{py:module} src.models.subnets.encoders.moe.moe_multi_head_attention_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEMultiHeadAttentionLayer <src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer
    :summary:
    ```
````

### API

`````{py:class} MoEMultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range, connection_type='skip', expansion_rate=4, num_experts=4, k=2, noisy_gating=True)
:canonical: src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer.forward
```

````

`````
