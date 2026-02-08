# {py:mod}`src.models.subnets.modules.gated_graph_convolution`

```{py:module} src.models.subnets.modules.gated_graph_convolution
```

```{autodoc2-docstring} src.models.subnets.modules.gated_graph_convolution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GatedGraphConvolution <src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution>`
  - ```{autodoc2-docstring} src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution
    :summary:
    ```
````

### API

`````{py:class} GatedGraphConvolution(hidden_dim: int, aggregation: str = 'sum', norm: str = 'batch', activation: str = 'relu', learn_affine: bool = True, gated: bool = True, bias: bool = True)
:canonical: src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution.__init__
```

````{py:method} reset_parameters()
:canonical: src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution.reset_parameters

```{autodoc2-docstring} src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution.reset_parameters
```

````

````{py:method} forward(h, e, mask)
:canonical: src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution.forward

```{autodoc2-docstring} src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution.forward
```

````

````{py:method} aggregate(Vh, mask, gates)
:canonical: src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution.aggregate

```{autodoc2-docstring} src.models.subnets.modules.gated_graph_convolution.GatedGraphConvolution.aggregate
```

````

`````
