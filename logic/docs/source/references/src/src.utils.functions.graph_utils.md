# {py:mod}`src.utils.functions.graph_utils`

```{py:module} src.utils.functions.graph_utils
```

```{autodoc2-docstring} src.utils.functions.graph_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`generate_adj_matrix <src.utils.functions.graph_utils.generate_adj_matrix>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.generate_adj_matrix
    :summary:
    ```
* - {py:obj}`get_edge_idx_dist <src.utils.functions.graph_utils.get_edge_idx_dist>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.get_edge_idx_dist
    :summary:
    ```
* - {py:obj}`sort_by_pairs <src.utils.functions.graph_utils.sort_by_pairs>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.sort_by_pairs
    :summary:
    ```
* - {py:obj}`get_adj_knn <src.utils.functions.graph_utils.get_adj_knn>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.get_adj_knn
    :summary:
    ```
* - {py:obj}`adj_to_idx <src.utils.functions.graph_utils.adj_to_idx>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.adj_to_idx
    :summary:
    ```
* - {py:obj}`idx_to_adj <src.utils.functions.graph_utils.idx_to_adj>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.idx_to_adj
    :summary:
    ```
* - {py:obj}`tour_to_adj <src.utils.functions.graph_utils.tour_to_adj>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.tour_to_adj
    :summary:
    ```
* - {py:obj}`get_adj_osm <src.utils.functions.graph_utils.get_adj_osm>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.get_adj_osm
    :summary:
    ```
* - {py:obj}`find_longest_path <src.utils.functions.graph_utils.find_longest_path>`
  - ```{autodoc2-docstring} src.utils.functions.graph_utils.find_longest_path
    :summary:
    ```
````

### API

````{py:function} generate_adj_matrix(size, num_edges, undirected=False, add_depot=True, negative=False)
:canonical: src.utils.functions.graph_utils.generate_adj_matrix

```{autodoc2-docstring} src.utils.functions.graph_utils.generate_adj_matrix
```
````

````{py:function} get_edge_idx_dist(dist_matrix, num_edges, add_depot=True, undirected=True)
:canonical: src.utils.functions.graph_utils.get_edge_idx_dist

```{autodoc2-docstring} src.utils.functions.graph_utils.get_edge_idx_dist
```
````

````{py:function} sort_by_pairs(graph_size, edge_idx)
:canonical: src.utils.functions.graph_utils.sort_by_pairs

```{autodoc2-docstring} src.utils.functions.graph_utils.sort_by_pairs
```
````

````{py:function} get_adj_knn(dist_mat, k_neighbors, add_depot=True, negative=True)
:canonical: src.utils.functions.graph_utils.get_adj_knn

```{autodoc2-docstring} src.utils.functions.graph_utils.get_adj_knn
```
````

````{py:function} adj_to_idx(adj_matrix, negative=True)
:canonical: src.utils.functions.graph_utils.adj_to_idx

```{autodoc2-docstring} src.utils.functions.graph_utils.adj_to_idx
```
````

````{py:function} idx_to_adj(edge_idx, negative=False)
:canonical: src.utils.functions.graph_utils.idx_to_adj

```{autodoc2-docstring} src.utils.functions.graph_utils.idx_to_adj
```
````

````{py:function} tour_to_adj(tour_nodes)
:canonical: src.utils.functions.graph_utils.tour_to_adj

```{autodoc2-docstring} src.utils.functions.graph_utils.tour_to_adj
```
````

````{py:function} get_adj_osm(coords, size, args, add_depot=True, negative=True)
:canonical: src.utils.functions.graph_utils.get_adj_osm

```{autodoc2-docstring} src.utils.functions.graph_utils.get_adj_osm
```
````

````{py:function} find_longest_path(dist_matrix, start_vertex=0)
:canonical: src.utils.functions.graph_utils.find_longest_path

```{autodoc2-docstring} src.utils.functions.graph_utils.find_longest_path
```
````
