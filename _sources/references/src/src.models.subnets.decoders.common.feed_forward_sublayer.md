# {py:mod}`src.models.subnets.decoders.common.feed_forward_sublayer`

```{py:module} src.models.subnets.decoders.common.feed_forward_sublayer
```

```{autodoc2-docstring} src.models.subnets.decoders.common.feed_forward_sublayer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FeedForwardSubLayer <src.models.subnets.decoders.common.feed_forward_sublayer.FeedForwardSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.common.feed_forward_sublayer.FeedForwardSubLayer
    :summary:
    ```
````

### API

`````{py:class} FeedForwardSubLayer(embed_dim: int, feed_forward_hidden: int, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, bias: bool = True, **kwargs)
:canonical: src.models.subnets.decoders.common.feed_forward_sublayer.FeedForwardSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.common.feed_forward_sublayer.FeedForwardSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.common.feed_forward_sublayer.FeedForwardSubLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.decoders.common.feed_forward_sublayer.FeedForwardSubLayer.forward

```{autodoc2-docstring} src.models.subnets.decoders.common.feed_forward_sublayer.FeedForwardSubLayer.forward
```

````

`````
