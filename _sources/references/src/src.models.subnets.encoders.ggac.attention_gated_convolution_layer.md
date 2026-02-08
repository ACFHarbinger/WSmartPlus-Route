# {py:mod}`src.models.subnets.encoders.ggac.attention_gated_convolution_layer`

```{py:module} src.models.subnets.encoders.ggac.attention_gated_convolution_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionGatedConvolutionLayer <src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer
    :summary:
    ```
````

### API

`````{py:class} AttentionGatedConvolutionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range, gated=True, agg='sum', bias=True)
:canonical: src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer.__init__
```

````{py:method} forward(h, e, mask=None)
:canonical: src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer.forward
```

````

`````
