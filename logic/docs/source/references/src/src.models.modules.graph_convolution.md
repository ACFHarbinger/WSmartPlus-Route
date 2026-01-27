# {py:mod}`src.models.modules.graph_convolution`

```{py:module} src.models.modules.graph_convolution
```

```{autodoc2-docstring} src.models.modules.graph_convolution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphConvolution <src.models.modules.graph_convolution.GraphConvolution>`
  - ```{autodoc2-docstring} src.models.modules.graph_convolution.GraphConvolution
    :summary:
    ```
````

### API

`````{py:class} GraphConvolution(in_channels: int, out_channels: int, aggregation: str = 'sum', bias: bool = True)
:canonical: src.models.modules.graph_convolution.GraphConvolution

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.graph_convolution.GraphConvolution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.graph_convolution.GraphConvolution.__init__
```

````{py:method} init_parameters()
:canonical: src.models.modules.graph_convolution.GraphConvolution.init_parameters

```{autodoc2-docstring} src.models.modules.graph_convolution.GraphConvolution.init_parameters
```

````

````{py:method} forward(h, mask)
:canonical: src.models.modules.graph_convolution.GraphConvolution.forward

```{autodoc2-docstring} src.models.modules.graph_convolution.GraphConvolution.forward
```

````

````{py:method} single_graph_forward(h, adj)
:canonical: src.models.modules.graph_convolution.GraphConvolution.single_graph_forward

```{autodoc2-docstring} src.models.modules.graph_convolution.GraphConvolution.single_graph_forward
```

````

`````
