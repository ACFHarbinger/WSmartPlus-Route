# {py:mod}`src.policies.helpers.operators.destroy.worst`

```{py:module} src.policies.helpers.operators.destroy.worst
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy.worst
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`worst_removal <src.policies.helpers.operators.destroy.worst.worst_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.worst.worst_removal
    :summary:
    ```
* - {py:obj}`worst_profit_removal <src.policies.helpers.operators.destroy.worst.worst_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.worst.worst_profit_removal
    :summary:
    ```
````

### API

````{py:function} worst_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, p: float = 1.0, rng: typing.Optional[numpy.random.Generator] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.worst.worst_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.worst.worst_removal
```
````

````{py:function} worst_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, p: float = 1.0, rng: typing.Optional[numpy.random.Generator] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.worst.worst_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.worst.worst_profit_removal
```
````
