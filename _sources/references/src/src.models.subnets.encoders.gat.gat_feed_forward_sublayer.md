# {py:mod}`src.models.subnets.encoders.gat.gat_feed_forward_sublayer`

```{py:module} src.models.subnets.encoders.gat.gat_feed_forward_sublayer
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_feed_forward_sublayer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GATFeedForwardSubLayer <src.models.subnets.encoders.gat.gat_feed_forward_sublayer.GATFeedForwardSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_feed_forward_sublayer.GATFeedForwardSubLayer
    :summary:
    ```
````

### API

`````{py:class} GATFeedForwardSubLayer(embed_dim: int, feed_forward_hidden: int, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, bias: bool = True, **kwargs)
:canonical: src.models.subnets.encoders.gat.gat_feed_forward_sublayer.GATFeedForwardSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_feed_forward_sublayer.GATFeedForwardSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_feed_forward_sublayer.GATFeedForwardSubLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.gat.gat_feed_forward_sublayer.GATFeedForwardSubLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.gat.gat_feed_forward_sublayer.GATFeedForwardSubLayer.forward
```

````

`````
