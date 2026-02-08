# {py:mod}`src.models.subnets.encoders.tgc.conv_layer`

```{py:module} src.models.subnets.encoders.tgc.conv_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphConvolutionLayer <src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer
    :summary:
    ```
````

### API

`````{py:class} GraphConvolutionLayer(embed_dim, feed_forward_hidden, agg, normalization, epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups, activation, af_param, threshold, replacement_value, n_params, uniform_range)
:canonical: src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer.__init__
```

````{py:method} forward(h, mask)
:canonical: src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer.forward
```

````

`````
