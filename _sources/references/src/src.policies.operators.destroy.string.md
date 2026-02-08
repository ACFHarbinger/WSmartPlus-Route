# {py:mod}`src.policies.operators.destroy.string`

```{py:module} src.policies.operators.destroy.string
```

```{autodoc2-docstring} src.policies.operators.destroy.string
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`string_removal <src.policies.operators.destroy.string.string_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy.string.string_removal
    :summary:
    ```
* - {py:obj}`_propagate_string_removal <src.policies.operators.destroy.string._propagate_string_removal>`
  - ```{autodoc2-docstring} src.policies.operators.destroy.string._propagate_string_removal
    :summary:
    ```
````

### API

````{py:function} string_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, max_string_len: int = 5, avg_string_len: float = 3.0) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.operators.destroy.string.string_removal

```{autodoc2-docstring} src.policies.operators.destroy.string.string_removal
```
````

````{py:function} _propagate_string_removal(routes: typing.List[typing.List[int]], removed: typing.List[int], dist_matrix: numpy.ndarray, seed_nodes: typing.List[int], n_remove: int, max_string_len: int) -> None
:canonical: src.policies.operators.destroy.string._propagate_string_removal

```{autodoc2-docstring} src.policies.operators.destroy.string._propagate_string_removal
```
````
