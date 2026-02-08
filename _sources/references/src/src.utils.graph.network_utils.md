# {py:mod}`src.utils.graph.network_utils`

```{py:module} src.utils.graph.network_utils
```

```{autodoc2-docstring} src.utils.graph.network_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_edges <src.utils.graph.network_utils.apply_edges>`
  - ```{autodoc2-docstring} src.utils.graph.network_utils.apply_edges
    :summary:
    ```
* - {py:obj}`get_paths_between_states <src.utils.graph.network_utils.get_paths_between_states>`
  - ```{autodoc2-docstring} src.utils.graph.network_utils.get_paths_between_states
    :summary:
    ```
````

### API

````{py:function} apply_edges(dist_matrix: numpy.ndarray, edge_thresh: float, edge_method: typing.Optional[str]) -> typing.Tuple[numpy.ndarray, typing.Optional[typing.Dict[typing.Tuple[int, int], typing.List[int]]], typing.Optional[numpy.ndarray]]
:canonical: src.utils.graph.network_utils.apply_edges

```{autodoc2-docstring} src.utils.graph.network_utils.apply_edges
```
````

````{py:function} get_paths_between_states(n_bins: int, shortest_paths: typing.Optional[typing.Dict[typing.Tuple[int, int], typing.List[int]]] = None) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.utils.graph.network_utils.get_paths_between_states

```{autodoc2-docstring} src.utils.graph.network_utils.get_paths_between_states
```
````
