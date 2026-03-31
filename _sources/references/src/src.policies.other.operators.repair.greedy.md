# {py:mod}`src.policies.other.operators.repair.greedy`

```{py:module} src.policies.other.operators.repair.greedy
```

```{autodoc2-docstring} src.policies.other.operators.repair.greedy
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`greedy_insertion <src.policies.other.operators.repair.greedy.greedy_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.greedy.greedy_insertion
    :summary:
    ```
* - {py:obj}`greedy_profit_insertion <src.policies.other.operators.repair.greedy.greedy_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.greedy.greedy_profit_insertion
    :summary:
    ```
````

### API

````{py:function} greedy_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True, noise: float = 0.0) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.greedy.greedy_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.greedy.greedy_insertion
```
````

````{py:function} greedy_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False, noise: float = 0.0) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.greedy.greedy_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.greedy.greedy_profit_insertion
```
````
