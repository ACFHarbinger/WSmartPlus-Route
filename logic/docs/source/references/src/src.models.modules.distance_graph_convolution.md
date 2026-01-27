# {py:mod}`src.models.modules.distance_graph_convolution`

```{py:module} src.models.modules.distance_graph_convolution
```

```{autodoc2-docstring} src.models.modules.distance_graph_convolution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistanceAwareGraphConvolution <src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution>`
  - ```{autodoc2-docstring} src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution
    :summary:
    ```
````

### API

`````{py:class} DistanceAwareGraphConvolution(in_channels: int, out_channels: int, distance_influence: str = 'inverse', aggregation: str = 'sum', bias: bool = True)
:canonical: src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.__init__
```

````{py:method} init_parameters()
:canonical: src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.init_parameters

```{autodoc2-docstring} src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.init_parameters
```

````

````{py:method} get_distance_weights(dist_matrix)
:canonical: src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.get_distance_weights

```{autodoc2-docstring} src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.get_distance_weights
```

````

````{py:method} forward(h, adj, dist_matrix=None)
:canonical: src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.forward

```{autodoc2-docstring} src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.forward
```

````

````{py:method} single_graph_forward(h, adj, dist_matrix=None)
:canonical: src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.single_graph_forward

```{autodoc2-docstring} src.models.modules.distance_graph_convolution.DistanceAwareGraphConvolution.single_graph_forward
```

````

`````
