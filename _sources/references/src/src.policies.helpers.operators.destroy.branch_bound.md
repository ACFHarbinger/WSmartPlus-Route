# {py:mod}`src.policies.helpers.operators.destroy.branch_bound`

```{py:module} src.policies.helpers.operators.destroy.branch_bound
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_marginal_detour <src.policies.helpers.operators.destroy.branch_bound._marginal_detour>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._marginal_detour
    :summary:
    ```
* - {py:obj}`_recovery_insertion_cost <src.policies.helpers.operators.destroy.branch_bound._recovery_insertion_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._recovery_insertion_cost
    :summary:
    ```
* - {py:obj}`_profit_contribution <src.policies.helpers.operators.destroy.branch_bound._profit_contribution>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._profit_contribution
    :summary:
    ```
* - {py:obj}`_lds_remove <src.policies.helpers.operators.destroy.branch_bound._lds_remove>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._lds_remove
    :summary:
    ```
* - {py:obj}`_lds_remove_profit <src.policies.helpers.operators.destroy.branch_bound._lds_remove_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._lds_remove_profit
    :summary:
    ```
* - {py:obj}`bb_removal <src.policies.helpers.operators.destroy.branch_bound.bb_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound.bb_removal
    :summary:
    ```
* - {py:obj}`bb_profit_removal <src.policies.helpers.operators.destroy.branch_bound.bb_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound.bb_profit_removal
    :summary:
    ```
````

### API

````{py:function} _marginal_detour(node: int, r_idx: int, pos: int, routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.helpers.operators.destroy.branch_bound._marginal_detour

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._marginal_detour
```
````

````{py:function} _recovery_insertion_cost(node: int, routes: typing.List[typing.List[int]], loads: typing.List[float], wastes: typing.Dict[int, float], dist_matrix: numpy.ndarray, capacity: float) -> float
:canonical: src.policies.helpers.operators.destroy.branch_bound._recovery_insertion_cost

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._recovery_insertion_cost
```
````

````{py:function} _profit_contribution(node: int, r_idx: int, pos: int, routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float) -> float
:canonical: src.policies.helpers.operators.destroy.branch_bound._profit_contribution

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._profit_contribution
```
````

````{py:function} _lds_remove(routes: typing.List[typing.List[int]], loads: typing.List[float], node_positions: typing.Dict[int, typing.Tuple[int, int]], candidates: typing.List[typing.Tuple[float, int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, discrep: int, current_removal: typing.List[int], current_savings: float, best_state: typing.List[typing.Optional[typing.Tuple[float, typing.List[int]]]]) -> None
:canonical: src.policies.helpers.operators.destroy.branch_bound._lds_remove

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._lds_remove
```
````

````{py:function} _lds_remove_profit(routes: typing.List[typing.List[int]], loads: typing.List[float], node_positions: typing.Dict[int, typing.Tuple[int, int]], candidates: typing.List[typing.Tuple[float, int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, discrep: int, current_removal: typing.List[int], current_profit_freed: float, best_state: typing.List[typing.Optional[typing.Tuple[float, typing.List[int]]]]) -> None
:canonical: src.policies.helpers.operators.destroy.branch_bound._lds_remove_profit

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound._lds_remove_profit
```
````

````{py:function} bb_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, max_discrepancy: int = 1, rng: typing.Optional[random.Random] = None, noise: float = 0.0) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.branch_bound.bb_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound.bb_removal
```
````

````{py:function} bb_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, max_discrepancy: int = 1, rng: typing.Optional[random.Random] = None, noise: float = 0.0) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.branch_bound.bb_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.branch_bound.bb_profit_removal
```
````
