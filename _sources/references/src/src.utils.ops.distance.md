# {py:mod}`src.utils.ops.distance`

```{py:module} src.utils.ops.distance
```

```{autodoc2-docstring} src.utils.ops.distance
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_distance <src.utils.ops.distance.get_distance>`
  - ```{autodoc2-docstring} src.utils.ops.distance.get_distance
    :summary:
    ```
* - {py:obj}`get_distance_matrix <src.utils.ops.distance.get_distance_matrix>`
  - ```{autodoc2-docstring} src.utils.ops.distance.get_distance_matrix
    :summary:
    ```
* - {py:obj}`get_tour_length <src.utils.ops.distance.get_tour_length>`
  - ```{autodoc2-docstring} src.utils.ops.distance.get_tour_length
    :summary:
    ```
* - {py:obj}`get_open_tour_length <src.utils.ops.distance.get_open_tour_length>`
  - ```{autodoc2-docstring} src.utils.ops.distance.get_open_tour_length
    :summary:
    ```
````

### API

````{py:function} get_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
:canonical: src.utils.ops.distance.get_distance

```{autodoc2-docstring} src.utils.ops.distance.get_distance
```
````

````{py:function} get_distance_matrix(locs: torch.Tensor) -> torch.Tensor
:canonical: src.utils.ops.distance.get_distance_matrix

```{autodoc2-docstring} src.utils.ops.distance.get_distance_matrix
```
````

````{py:function} get_tour_length(ordered_locs: torch.Tensor) -> torch.Tensor
:canonical: src.utils.ops.distance.get_tour_length

```{autodoc2-docstring} src.utils.ops.distance.get_tour_length
```
````

````{py:function} get_open_tour_length(ordered_locs: torch.Tensor) -> torch.Tensor
:canonical: src.utils.ops.distance.get_open_tour_length

```{autodoc2-docstring} src.utils.ops.distance.get_open_tour_length
```
````
