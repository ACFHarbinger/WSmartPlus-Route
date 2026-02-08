# {py:mod}`src.utils.graph.generation`

```{py:module} src.utils.graph.generation
```

```{autodoc2-docstring} src.utils.graph.generation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`generate_adj_matrix <src.utils.graph.generation.generate_adj_matrix>`
  - ```{autodoc2-docstring} src.utils.graph.generation.generate_adj_matrix
    :summary:
    ```
* - {py:obj}`get_edge_idx_dist <src.utils.graph.generation.get_edge_idx_dist>`
  - ```{autodoc2-docstring} src.utils.graph.generation.get_edge_idx_dist
    :summary:
    ```
* - {py:obj}`get_adj_knn <src.utils.graph.generation.get_adj_knn>`
  - ```{autodoc2-docstring} src.utils.graph.generation.get_adj_knn
    :summary:
    ```
* - {py:obj}`get_adj_osm <src.utils.graph.generation.get_adj_osm>`
  - ```{autodoc2-docstring} src.utils.graph.generation.get_adj_osm
    :summary:
    ```
````

### API

````{py:function} generate_adj_matrix(size: int, num_edges: typing.Union[int, float], undirected: bool = False, add_depot: bool = True, negative: bool = False) -> numpy.ndarray
:canonical: src.utils.graph.generation.generate_adj_matrix

```{autodoc2-docstring} src.utils.graph.generation.generate_adj_matrix
```
````

````{py:function} get_edge_idx_dist(dist_matrix: numpy.ndarray, num_edges: typing.Union[int, float], add_depot: bool = True, undirected: bool = True) -> numpy.ndarray
:canonical: src.utils.graph.generation.get_edge_idx_dist

```{autodoc2-docstring} src.utils.graph.generation.get_edge_idx_dist
```
````

````{py:function} get_adj_knn(dist_mat: numpy.ndarray, k_neighbors: typing.Union[int, float], add_depot: bool = True, negative: bool = True) -> numpy.ndarray
:canonical: src.utils.graph.generation.get_adj_knn

```{autodoc2-docstring} src.utils.graph.generation.get_adj_knn
```
````

````{py:function} get_adj_osm(coords: typing.Any, size: int, args: list, add_depot: bool = True, negative: bool = True) -> numpy.ndarray
:canonical: src.utils.graph.generation.get_adj_osm

```{autodoc2-docstring} src.utils.graph.generation.get_adj_osm
```
````
