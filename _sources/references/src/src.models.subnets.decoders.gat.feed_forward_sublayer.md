# {py:mod}`src.models.subnets.decoders.gat.feed_forward_sublayer`

```{py:module} src.models.subnets.decoders.gat.feed_forward_sublayer
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.feed_forward_sublayer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FeedForwardSubLayer <src.models.subnets.decoders.gat.feed_forward_sublayer.FeedForwardSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.gat.feed_forward_sublayer.FeedForwardSubLayer
    :summary:
    ```
````

### API

`````{py:class} FeedForwardSubLayer(embed_dim, feed_forward_hidden, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True)
:canonical: src.models.subnets.decoders.gat.feed_forward_sublayer.FeedForwardSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.gat.feed_forward_sublayer.FeedForwardSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.feed_forward_sublayer.FeedForwardSubLayer.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.decoders.gat.feed_forward_sublayer.FeedForwardSubLayer.forward

```{autodoc2-docstring} src.models.subnets.decoders.gat.feed_forward_sublayer.FeedForwardSubLayer.forward
```

````

`````
