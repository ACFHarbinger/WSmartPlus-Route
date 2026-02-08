# {py:mod}`src.utils.graph.conversion`

```{py:module} src.utils.graph.conversion
```

```{autodoc2-docstring} src.utils.graph.conversion
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`adj_to_idx <src.utils.graph.conversion.adj_to_idx>`
  - ```{autodoc2-docstring} src.utils.graph.conversion.adj_to_idx
    :summary:
    ```
* - {py:obj}`idx_to_adj <src.utils.graph.conversion.idx_to_adj>`
  - ```{autodoc2-docstring} src.utils.graph.conversion.idx_to_adj
    :summary:
    ```
* - {py:obj}`tour_to_adj <src.utils.graph.conversion.tour_to_adj>`
  - ```{autodoc2-docstring} src.utils.graph.conversion.tour_to_adj
    :summary:
    ```
* - {py:obj}`sort_by_pairs <src.utils.graph.conversion.sort_by_pairs>`
  - ```{autodoc2-docstring} src.utils.graph.conversion.sort_by_pairs
    :summary:
    ```
````

### API

````{py:function} adj_to_idx(adj_matrix: numpy.ndarray, negative: bool = True) -> numpy.ndarray
:canonical: src.utils.graph.conversion.adj_to_idx

```{autodoc2-docstring} src.utils.graph.conversion.adj_to_idx
```
````

````{py:function} idx_to_adj(edge_idx: typing.Union[torch.Tensor, numpy.ndarray], negative: bool = False) -> numpy.ndarray
:canonical: src.utils.graph.conversion.idx_to_adj

```{autodoc2-docstring} src.utils.graph.conversion.idx_to_adj
```
````

````{py:function} tour_to_adj(tour_nodes: list) -> numpy.ndarray
:canonical: src.utils.graph.conversion.tour_to_adj

```{autodoc2-docstring} src.utils.graph.conversion.tour_to_adj
```
````

````{py:function} sort_by_pairs(graph_size: int, edge_idx: torch.Tensor) -> torch.Tensor
:canonical: src.utils.graph.conversion.sort_by_pairs

```{autodoc2-docstring} src.utils.graph.conversion.sort_by_pairs
```
````
