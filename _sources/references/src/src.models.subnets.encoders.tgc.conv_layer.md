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

`````{py:class} GraphConvolutionLayer(embed_dim: int, feed_forward_hidden: int, agg: str, normalization: str, epsilon_alpha: float, learn_affine: bool, track_stats: bool, mbeta: float, lr_k: float, n_groups: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, uniform_range: typing.List[float])
:canonical: src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer.__init__
```

````{py:method} forward(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.tgc.conv_layer.GraphConvolutionLayer.forward
```

````

`````
