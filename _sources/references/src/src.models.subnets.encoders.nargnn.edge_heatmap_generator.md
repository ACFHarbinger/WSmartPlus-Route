# {py:mod}`src.models.subnets.encoders.nargnn.edge_heatmap_generator`

```{py:module} src.models.subnets.encoders.nargnn.edge_heatmap_generator
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_heatmap_generator
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EdgeHeatmapGenerator <src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator
    :summary:
    ```
````

### API

`````{py:class} EdgeHeatmapGenerator(embed_dim: int, num_layers: int, act_fn: str | collections.abc.Callable = 'silu', linear_bias: bool = True, undirected_graph: bool = True)
:canonical: src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator.__init__
```

````{py:method} forward(graph: torch_geometric.data.Batch) -> torch.Tensor
:canonical: src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator.forward

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator.forward
```

````

````{py:method} _make_heatmap_logits(batch_graph: torch_geometric.data.Batch) -> torch.Tensor
:canonical: src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator._make_heatmap_logits

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_heatmap_generator.EdgeHeatmapGenerator._make_heatmap_logits
```

````

`````
