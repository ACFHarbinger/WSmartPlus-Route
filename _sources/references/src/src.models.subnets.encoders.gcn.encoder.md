# {py:mod}`src.models.subnets.encoders.gcn.encoder`

```{py:module} src.models.subnets.encoders.gcn.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.gcn.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphConvolutionEncoder <src.models.subnets.encoders.gcn.encoder.GraphConvolutionEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gcn.encoder.GraphConvolutionEncoder
    :summary:
    ```
````

### API

`````{py:class} GraphConvolutionEncoder(n_layers: int, feed_forward_hidden: int, agg: str = 'sum', norm: str = 'layer', learn_affine: bool = True, track_norm: bool = False, gated: bool = True, *args: typing.Any, **kwargs: typing.Any)
:canonical: src.models.subnets.encoders.gcn.encoder.GraphConvolutionEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.gcn.encoder.GraphConvolutionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gcn.encoder.GraphConvolutionEncoder.__init__
```

````{py:method} forward(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.encoders.gcn.encoder.GraphConvolutionEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.gcn.encoder.GraphConvolutionEncoder.forward
```

````

`````
