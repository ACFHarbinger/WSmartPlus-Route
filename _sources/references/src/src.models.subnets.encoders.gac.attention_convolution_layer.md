# {py:mod}`src.models.subnets.encoders.gac.attention_convolution_layer`

```{py:module} src.models.subnets.encoders.gac.attention_convolution_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionConvolutionLayer <src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer
    :summary:
    ```
````

### API

`````{py:class} AttentionConvolutionLayer(n_heads, embed_dim, feed_forward_hidden, agg, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range)
:canonical: src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer.__init__
```

````{py:method} forward(h, edges=None, mask=None)
:canonical: src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer.forward
```

````

`````
