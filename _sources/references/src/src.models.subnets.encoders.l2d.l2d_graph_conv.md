# {py:mod}`src.models.subnets.encoders.l2d.l2d_graph_conv`

```{py:module} src.models.subnets.encoders.l2d.l2d_graph_conv
```

```{autodoc2-docstring} src.models.subnets.encoders.l2d.l2d_graph_conv
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`L2DGraphConv <src.models.subnets.encoders.l2d.l2d_graph_conv.L2DGraphConv>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.l2d.l2d_graph_conv.L2DGraphConv
    :summary:
    ```
````

### API

`````{py:class} L2DGraphConv(embed_dim: int, hidden_dim: int = 128, aggregation: str = 'mean')
:canonical: src.models.subnets.encoders.l2d.l2d_graph_conv.L2DGraphConv

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.l2d.l2d_graph_conv.L2DGraphConv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.l2d.l2d_graph_conv.L2DGraphConv.__init__
```

````{py:method} forward(h: torch.Tensor, adj_prec: torch.Tensor, adj_mach: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.encoders.l2d.l2d_graph_conv.L2DGraphConv.forward

```{autodoc2-docstring} src.models.subnets.encoders.l2d.l2d_graph_conv.L2DGraphConv.forward
```

````

`````
