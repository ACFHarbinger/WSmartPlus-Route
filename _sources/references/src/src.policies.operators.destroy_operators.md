# {py:mod}`src.policies.operators.destroy_operators`

```{py:module} src.policies.operators.destroy_operators
```

```{autodoc2-docstring} src.policies.operators.destroy_operators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`random_removal <src.policies.operators.destroy_operators.random_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy_operators.random_removal
    :summary:
    ```
* - {py:obj}`worst_removal <src.policies.operators.destroy_operators.worst_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy_operators.worst_removal
    :summary:
    ```
* - {py:obj}`cluster_removal <src.policies.operators.destroy_operators.cluster_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy_operators.cluster_removal
    :summary:
    ```
* - {py:obj}`shaw_removal <src.policies.operators.destroy_operators.shaw_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy_operators.shaw_removal
    :summary:
    ```
* - {py:obj}`string_removal <src.policies.operators.destroy_operators.string_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy_operators.string_removal
    :summary:
    ```
* - {py:obj}`_propagate_string_removal <src.policies.operators.destroy_operators._propagate_string_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy_operators._propagate_string_removal
    :summary:
    ```
````

### API

````{py:function} random_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.operators.destroy_operators.random_removal

```{autodoc2-docstring} src.policies.operators.destroy_operators.random_removal
```
````

````{py:function} worst_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.operators.destroy_operators.worst_removal

```{autodoc2-docstring} src.policies.operators.destroy_operators.worst_removal
```
````

````{py:function} cluster_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, nodes: typing.List[int]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.operators.destroy_operators.cluster_removal

```{autodoc2-docstring} src.policies.operators.destroy_operators.cluster_removal
```
````

````{py:function} shaw_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, demands: typing.Optional[typing.Dict[int, float]] = None, time_windows: typing.Optional[typing.Dict[int, tuple]] = None, phi: float = 9.0, chi: float = 3.0, psi: float = 2.0, randomization_factor: float = 2.0) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.operators.destroy_operators.shaw_removal

```{autodoc2-docstring} src.policies.operators.destroy_operators.shaw_removal
```
````

````{py:function} string_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, max_string_len: int = 5, avg_string_len: float = 3.0) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.operators.destroy_operators.string_removal

```{autodoc2-docstring} src.policies.operators.destroy_operators.string_removal
```
````

````{py:function} _propagate_string_removal(routes: typing.List[typing.List[int]], removed: typing.List[int], dist_matrix: numpy.ndarray, seed_nodes: typing.List[int], n_remove: int, max_string_len: int) -> None
:canonical: src.policies.operators.destroy_operators._propagate_string_removal

```{autodoc2-docstring} src.policies.operators.destroy_operators._propagate_string_removal
```
````
