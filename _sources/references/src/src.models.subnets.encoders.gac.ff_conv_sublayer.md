# {py:mod}`src.models.subnets.encoders.gac.ff_conv_sublayer`

```{py:module} src.models.subnets.encoders.gac.ff_conv_sublayer
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.ff_conv_sublayer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FFConvSubLayer <src.models.subnets.encoders.gac.ff_conv_sublayer.FFConvSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gac.ff_conv_sublayer.FFConvSubLayer
    :summary:
    ```
````

### API

`````{py:class} FFConvSubLayer(embed_dim, feed_forward_hidden, agg, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True)
:canonical: src.models.subnets.encoders.gac.ff_conv_sublayer.FFConvSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.gac.ff_conv_sublayer.FFConvSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.ff_conv_sublayer.FFConvSubLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.encoders.gac.ff_conv_sublayer.FFConvSubLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.gac.ff_conv_sublayer.FFConvSubLayer.forward
```

````

`````
