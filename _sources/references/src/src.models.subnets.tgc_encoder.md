# {py:mod}`src.models.subnets.tgc_encoder`

```{py:module} src.models.subnets.tgc_encoder
```

```{autodoc2-docstring} src.models.subnets.tgc_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TGCFeedForwardSubLayer <src.models.subnets.tgc_encoder.TGCFeedForwardSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCFeedForwardSubLayer
    :summary:
    ```
* - {py:obj}`TGCMultiHeadAttentionLayer <src.models.subnets.tgc_encoder.TGCMultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCMultiHeadAttentionLayer
    :summary:
    ```
* - {py:obj}`FFConvSubLayer <src.models.subnets.tgc_encoder.FFConvSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.tgc_encoder.FFConvSubLayer
    :summary:
    ```
* - {py:obj}`GraphConvolutionLayer <src.models.subnets.tgc_encoder.GraphConvolutionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.tgc_encoder.GraphConvolutionLayer
    :summary:
    ```
* - {py:obj}`TransGraphConvEncoder <src.models.subnets.tgc_encoder.TransGraphConvEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.tgc_encoder.TransGraphConvEncoder
    :summary:
    ```
````

### API

`````{py:class} TGCFeedForwardSubLayer(embed_dim, feed_forward_hidden, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True)
:canonical: src.models.subnets.tgc_encoder.TGCFeedForwardSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCFeedForwardSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCFeedForwardSubLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.tgc_encoder.TGCFeedForwardSubLayer.forward

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCFeedForwardSubLayer.forward
```

````

`````

`````{py:class} TGCMultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range)
:canonical: src.models.subnets.tgc_encoder.TGCMultiHeadAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCMultiHeadAttentionLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.tgc_encoder.TGCMultiHeadAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TGCMultiHeadAttentionLayer.forward
```

````

`````

`````{py:class} FFConvSubLayer(embed_dim, feed_forward_hidden, agg, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True)
:canonical: src.models.subnets.tgc_encoder.FFConvSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.tgc_encoder.FFConvSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.tgc_encoder.FFConvSubLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.tgc_encoder.FFConvSubLayer.forward

```{autodoc2-docstring} src.models.subnets.tgc_encoder.FFConvSubLayer.forward
```

````

`````

`````{py:class} GraphConvolutionLayer(embed_dim, feed_forward_hidden, agg, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range)
:canonical: src.models.subnets.tgc_encoder.GraphConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.tgc_encoder.GraphConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.tgc_encoder.GraphConvolutionLayer.__init__
```

````{py:method} forward(h, mask)
:canonical: src.models.subnets.tgc_encoder.GraphConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.tgc_encoder.GraphConvolutionLayer.forward
```

````

`````

`````{py:class} TransGraphConvEncoder(n_heads, embed_dim, n_layers, n_sublayers=None, feed_forward_hidden=512, normalization='batch', epsilon_alpha=1e-05, learn_affine=True, track_stats=False, momentum_beta=0.1, locresp_k=1.0, n_groups=3, activation='gelu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=[0.125, 1 / 3], dropout_rate=0.1, agg='mean')
:canonical: src.models.subnets.tgc_encoder.TransGraphConvEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TransGraphConvEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TransGraphConvEncoder.__init__
```

````{py:method} forward(x, edges)
:canonical: src.models.subnets.tgc_encoder.TransGraphConvEncoder.forward

```{autodoc2-docstring} src.models.subnets.tgc_encoder.TransGraphConvEncoder.forward
```

````

`````
