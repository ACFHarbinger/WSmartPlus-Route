# {py:mod}`src.models.subnets.moe_encoder`

```{py:module} src.models.subnets.moe_encoder
```

```{autodoc2-docstring} src.models.subnets.moe_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEMultiHeadAttentionLayer <src.models.subnets.moe_encoder.MoEMultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEMultiHeadAttentionLayer
    :summary:
    ```
* - {py:obj}`MoEGraphAttentionEncoder <src.models.subnets.moe_encoder.MoEGraphAttentionEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEGraphAttentionEncoder
    :summary:
    ```
````

### API

`````{py:class} MoEMultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range, connection_type='skip', expansion_rate=4, num_experts=4, k=2, noisy_gating=True)
:canonical: src.models.subnets.moe_encoder.MoEMultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEMultiHeadAttentionLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.moe_encoder.MoEMultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEMultiHeadAttentionLayer.forward
```

````

`````

`````{py:class} MoEGraphAttentionEncoder(n_heads, embed_dim, n_layers, n_sublayers=None, feed_forward_hidden=512, normalization='batch', epsilon_alpha=1e-05, learn_affine=True, track_stats=False, momentum_beta=0.1, locresp_k=1.0, n_groups=3, activation='gelu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=[0.125, 1 / 3], dropout_rate=0.1, agg=None, connection_type='skip', expansion_rate=4, num_experts=4, k=2, noisy_gating=True)
:canonical: src.models.subnets.moe_encoder.MoEGraphAttentionEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEGraphAttentionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEGraphAttentionEncoder.__init__
```

````{py:method} forward(x, edges=None)
:canonical: src.models.subnets.moe_encoder.MoEGraphAttentionEncoder.forward

```{autodoc2-docstring} src.models.subnets.moe_encoder.MoEGraphAttentionEncoder.forward
```

````

`````
