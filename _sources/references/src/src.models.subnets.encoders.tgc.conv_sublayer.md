# {py:mod}`src.models.subnets.encoders.tgc.conv_sublayer`

```{py:module} src.models.subnets.encoders.tgc.conv_sublayer
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_sublayer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FFConvSubLayer <src.models.subnets.encoders.tgc.conv_sublayer.FFConvSubLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_sublayer.FFConvSubLayer
    :summary:
    ```
````

### API

`````{py:class} FFConvSubLayer(embed_dim: int, feed_forward_hidden: int, agg: str, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, dist_range: typing.List[float], bias: bool = True)
:canonical: src.models.subnets.encoders.tgc.conv_sublayer.FFConvSubLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_sublayer.FFConvSubLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_sublayer.FFConvSubLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.tgc.conv_sublayer.FFConvSubLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_sublayer.FFConvSubLayer.forward
```

````

`````
