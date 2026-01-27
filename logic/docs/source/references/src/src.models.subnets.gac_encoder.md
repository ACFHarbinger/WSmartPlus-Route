# {py:mod}`src.models.subnets.gac_encoder`

```{py:module} src.models.subnets.gac_encoder
```

```{autodoc2-docstring} src.models.subnets.gac_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FFConvSubLayer <src.models.subnets.gac_encoder.FFConvSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.gac_encoder.FFConvSubLayer
    :summary:
    ```
* - {py:obj}`AttentionConvolutionLayer <src.models.subnets.gac_encoder.AttentionConvolutionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.gac_encoder.AttentionConvolutionLayer
    :summary:
    ```
* - {py:obj}`GraphAttConvEncoder <src.models.subnets.gac_encoder.GraphAttConvEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.gac_encoder.GraphAttConvEncoder
    :summary:
    ```
````

### API

`````{py:class} FFConvSubLayer(embed_dim, feed_forward_hidden, agg, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True)
:canonical: src.models.subnets.gac_encoder.FFConvSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gac_encoder.FFConvSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gac_encoder.FFConvSubLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.gac_encoder.FFConvSubLayer.forward

```{autodoc2-docstring} src.models.subnets.gac_encoder.FFConvSubLayer.forward
```

````

`````

`````{py:class} AttentionConvolutionLayer(n_heads, embed_dim, feed_forward_hidden, agg, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range)
:canonical: src.models.subnets.gac_encoder.AttentionConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gac_encoder.AttentionConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gac_encoder.AttentionConvolutionLayer.__init__
```

````{py:method} forward(h, mask)
:canonical: src.models.subnets.gac_encoder.AttentionConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.gac_encoder.AttentionConvolutionLayer.forward
```

````

`````

`````{py:class} GraphAttConvEncoder(n_heads, embed_dim, n_layers, n_sublayers=None, feed_forward_hidden=512, normalization='batch', epsilon_alpha=1e-05, learn_affine=True, track_stats=False, momentum_beta=0.1, locresp_k=1.0, n_groups=3, activation='gelu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=[0.125, 1 / 3], dropout_rate=0.1, aggregate: str = 'sum')
:canonical: src.models.subnets.gac_encoder.GraphAttConvEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gac_encoder.GraphAttConvEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gac_encoder.GraphAttConvEncoder.__init__
```

````{py:method} forward(x, edges)
:canonical: src.models.subnets.gac_encoder.GraphAttConvEncoder.forward

```{autodoc2-docstring} src.models.subnets.gac_encoder.GraphAttConvEncoder.forward
```

````

`````
