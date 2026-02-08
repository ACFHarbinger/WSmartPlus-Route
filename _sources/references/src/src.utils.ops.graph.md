# {py:mod}`src.utils.ops.graph`

```{py:module} src.utils.ops.graph
```

```{autodoc2-docstring} src.utils.ops.graph
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sparsify_graph <src.utils.ops.graph.sparsify_graph>`
  - ```{autodoc2-docstring} src.utils.ops.graph.sparsify_graph
    :summary:
    ```
* - {py:obj}`_cached_full_graph_edge_index <src.utils.ops.graph._cached_full_graph_edge_index>`
  - ```{autodoc2-docstring} src.utils.ops.graph._cached_full_graph_edge_index
    :summary:
    ```
* - {py:obj}`get_full_graph_edge_index <src.utils.ops.graph.get_full_graph_edge_index>`
  - ```{autodoc2-docstring} src.utils.ops.graph.get_full_graph_edge_index
    :summary:
    ```
````

### API

````{py:function} sparsify_graph(cost_matrix: torch.Tensor, k_sparse: int, self_loop: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.utils.ops.graph.sparsify_graph

```{autodoc2-docstring} src.utils.ops.graph.sparsify_graph
```
````

````{py:function} _cached_full_graph_edge_index(num_node: int, self_loop: bool) -> torch.Tensor
:canonical: src.utils.ops.graph._cached_full_graph_edge_index

```{autodoc2-docstring} src.utils.ops.graph._cached_full_graph_edge_index
```
````

````{py:function} get_full_graph_edge_index(num_node: int, self_loop: bool = False) -> torch.Tensor
:canonical: src.utils.ops.graph.get_full_graph_edge_index

```{autodoc2-docstring} src.utils.ops.graph.get_full_graph_edge_index
```
````
