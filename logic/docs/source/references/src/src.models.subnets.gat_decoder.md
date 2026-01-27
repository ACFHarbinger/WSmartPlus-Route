# {py:mod}`src.models.subnets.gat_decoder`

```{py:module} src.models.subnets.gat_decoder
```

```{autodoc2-docstring} src.models.subnets.gat_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FeedForwardSubLayer <src.models.subnets.gat_decoder.FeedForwardSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.gat_decoder.FeedForwardSubLayer
    :summary:
    ```
* - {py:obj}`MultiHeadAttentionLayer <src.models.subnets.gat_decoder.MultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.gat_decoder.MultiHeadAttentionLayer
    :summary:
    ```
* - {py:obj}`GraphAttentionDecoder <src.models.subnets.gat_decoder.GraphAttentionDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.gat_decoder.GraphAttentionDecoder
    :summary:
    ```
````

### API

`````{py:class} FeedForwardSubLayer(embed_dim, feed_forward_hidden, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True)
:canonical: src.models.subnets.gat_decoder.FeedForwardSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gat_decoder.FeedForwardSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gat_decoder.FeedForwardSubLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.gat_decoder.FeedForwardSubLayer.forward

```{autodoc2-docstring} src.models.subnets.gat_decoder.FeedForwardSubLayer.forward
```

````

`````

`````{py:class} MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range)
:canonical: src.models.subnets.gat_decoder.MultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gat_decoder.MultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gat_decoder.MultiHeadAttentionLayer.__init__
```

````{py:method} forward(q, h, mask)
:canonical: src.models.subnets.gat_decoder.MultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.gat_decoder.MultiHeadAttentionLayer.forward
```

````

`````

`````{py:class} GraphAttentionDecoder(n_heads, embed_dim, n_layers, feed_forward_hidden=512, normalization='batch', epsilon_alpha=1e-05, learn_affine=True, track_stats=False, momentum_beta=0.1, locresp_k=1.0, n_groups=3, activation='gelu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=[0.125, 1 / 3], dropout_rate=0.1)
:canonical: src.models.subnets.gat_decoder.GraphAttentionDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gat_decoder.GraphAttentionDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gat_decoder.GraphAttentionDecoder.__init__
```

````{py:method} forward(q, h=None, mask=None)
:canonical: src.models.subnets.gat_decoder.GraphAttentionDecoder.forward

```{autodoc2-docstring} src.models.subnets.gat_decoder.GraphAttentionDecoder.forward
```

````

`````
