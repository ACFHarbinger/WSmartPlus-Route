# {py:mod}`src.policies.other.operators.repair.regret`

```{py:module} src.policies.other.operators.repair.regret
```

```{autodoc2-docstring} src.policies.other.operators.repair.regret
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`regret_2_insertion <src.policies.other.operators.repair.regret.regret_2_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.regret.regret_2_insertion
    :summary:
    ```
* - {py:obj}`regret_k_insertion <src.policies.other.operators.repair.regret.regret_k_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.regret.regret_k_insertion
    :summary:
    ```
* - {py:obj}`_get_insertion_options_with_profit <src.policies.other.operators.repair.regret._get_insertion_options_with_profit>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.regret._get_insertion_options_with_profit
    :summary:
    ```
* - {py:obj}`regret_profit_insertion <src.policies.other.operators.repair.regret.regret_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.regret.regret_profit_insertion
    :summary:
    ```
````

### API

````{py:function} regret_2_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: typing.Optional[float] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None, cost_unit: float = 1.0, expand_pool: bool = True, noise: float = 0.0) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.regret.regret_2_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.regret.regret_2_insertion
```
````

````{py:function} regret_k_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, k: int = 2, R: typing.Optional[float] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None, cost_unit: float = 1.0, expand_pool: bool = True, noise: float = 0.0) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.regret.regret_k_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.regret.regret_k_insertion
```
````

````{py:function} _get_insertion_options_with_profit(node: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, is_mandatory: bool, noise: float, max_dist: float) -> typing.List[typing.Tuple[float, int, int]]
:canonical: src.policies.other.operators.repair.regret._get_insertion_options_with_profit

```{autodoc2-docstring} src.policies.other.operators.repair.regret._get_insertion_options_with_profit
```
````

````{py:function} regret_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, k: int = 2, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False, noise: float = 0.0) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.regret.regret_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.regret.regret_profit_insertion
```
````
