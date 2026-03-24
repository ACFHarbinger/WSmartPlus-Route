# {py:mod}`src.policies.other.operators.repair.savings`

```{py:module} src.policies.other.operators.repair.savings
```

```{autodoc2-docstring} src.policies.other.operators.repair.savings
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`savings_insertion <src.policies.other.operators.repair.savings.savings_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.savings.savings_insertion
    :summary:
    ```
* - {py:obj}`savings_profit_insertion <src.policies.other.operators.repair.savings.savings_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.savings.savings_profit_insertion
    :summary:
    ```
````

### API

````{py:function} savings_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.savings.savings_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.savings.savings_insertion
```
````

````{py:function} savings_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, depot: int = 0, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.savings.savings_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.savings.savings_profit_insertion
```
````
