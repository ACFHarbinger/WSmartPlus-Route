# {py:mod}`src.policies.other.operators.repair.farthest`

```{py:module} src.policies.other.operators.repair.farthest
```

```{autodoc2-docstring} src.policies.other.operators.repair.farthest
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`farthest_insertion <src.policies.other.operators.repair.farthest.farthest_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.farthest.farthest_insertion
    :summary:
    ```
* - {py:obj}`farthest_profit_insertion <src.policies.other.operators.repair.farthest.farthest_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.farthest.farthest_profit_insertion
    :summary:
    ```
````

### API

````{py:function} farthest_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: typing.Optional[float] = None, C: typing.Optional[float] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.farthest.farthest_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.farthest.farthest_insertion
```
````

````{py:function} farthest_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.farthest.farthest_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.farthest.farthest_profit_insertion
```
````
