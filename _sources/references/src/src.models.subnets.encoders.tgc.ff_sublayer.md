# {py:mod}`src.models.subnets.encoders.tgc.ff_sublayer`

```{py:module} src.models.subnets.encoders.tgc.ff_sublayer
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.ff_sublayer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TGCFeedForwardSubLayer <src.models.subnets.encoders.tgc.ff_sublayer.TGCFeedForwardSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.tgc.ff_sublayer.TGCFeedForwardSubLayer
    :summary:
    ```
````

### API

`````{py:class} TGCFeedForwardSubLayer(embed_dim, feed_forward_hidden, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True)
:canonical: src.models.subnets.encoders.tgc.ff_sublayer.TGCFeedForwardSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.tgc.ff_sublayer.TGCFeedForwardSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.ff_sublayer.TGCFeedForwardSubLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.encoders.tgc.ff_sublayer.TGCFeedForwardSubLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.tgc.ff_sublayer.TGCFeedForwardSubLayer.forward
```

````

`````
