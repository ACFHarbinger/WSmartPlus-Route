# {py:mod}`src.policies.alns_aux.destroy_operators`

```{py:module} src.policies.alns_aux.destroy_operators
```

```{autodoc2-docstring} src.policies.alns_aux.destroy_operators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`random_removal <src.policies.alns_aux.destroy_operators.random_removal>`
  - ```{autodoc2-docstring} src.policies.alns_aux.destroy_operators.random_removal
    :summary:
    ```
* - {py:obj}`worst_removal <src.policies.alns_aux.destroy_operators.worst_removal>`
  - ```{autodoc2-docstring} src.policies.alns_aux.destroy_operators.worst_removal
    :summary:
    ```
* - {py:obj}`cluster_removal <src.policies.alns_aux.destroy_operators.cluster_removal>`
  - ```{autodoc2-docstring} src.policies.alns_aux.destroy_operators.cluster_removal
    :summary:
    ```
````

### API

````{py:function} random_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.alns_aux.destroy_operators.random_removal

```{autodoc2-docstring} src.policies.alns_aux.destroy_operators.random_removal
```
````

````{py:function} worst_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.alns_aux.destroy_operators.worst_removal

```{autodoc2-docstring} src.policies.alns_aux.destroy_operators.worst_removal
```
````

````{py:function} cluster_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, nodes: typing.List[int]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.alns_aux.destroy_operators.cluster_removal

```{autodoc2-docstring} src.policies.alns_aux.destroy_operators.cluster_removal
```
````
