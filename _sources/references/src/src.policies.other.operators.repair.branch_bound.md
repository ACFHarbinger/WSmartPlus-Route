# {py:mod}`src.policies.other.operators.repair.branch_bound`

```{py:module} src.policies.other.operators.repair.branch_bound
```

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compute_plan_cost <src.policies.other.operators.repair.branch_bound._compute_plan_cost>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._compute_plan_cost
    :summary:
    ```
* - {py:obj}`_get_sorted_positions <src.policies.other.operators.repair.branch_bound._get_sorted_positions>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._get_sorted_positions
    :summary:
    ```
* - {py:obj}`_get_sorted_positions_profit <src.policies.other.operators.repair.branch_bound._get_sorted_positions_profit>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._get_sorted_positions_profit
    :summary:
    ```
* - {py:obj}`_lds_reinsert <src.policies.other.operators.repair.branch_bound._lds_reinsert>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._lds_reinsert
    :summary:
    ```
* - {py:obj}`_lds_reinsert_profit <src.policies.other.operators.repair.branch_bound._lds_reinsert_profit>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._lds_reinsert_profit
    :summary:
    ```
* - {py:obj}`bb_insertion <src.policies.other.operators.repair.branch_bound.bb_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound.bb_insertion
    :summary:
    ```
* - {py:obj}`bb_profit_insertion <src.policies.other.operators.repair.branch_bound.bb_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound.bb_profit_insertion
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_Position <src.policies.other.operators.repair.branch_bound._Position>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._Position
    :summary:
    ```
* - {py:obj}`_Plan <src.policies.other.operators.repair.branch_bound._Plan>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._Plan
    :summary:
    ```
* - {py:obj}`_Loads <src.policies.other.operators.repair.branch_bound._Loads>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._Loads
    :summary:
    ```
````

### API

````{py:data} _Position
:canonical: src.policies.other.operators.repair.branch_bound._Position
:value: >
   None

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._Position
```

````

````{py:data} _Plan
:canonical: src.policies.other.operators.repair.branch_bound._Plan
:value: >
   None

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._Plan
```

````

````{py:data} _Loads
:canonical: src.policies.other.operators.repair.branch_bound._Loads
:value: >
   None

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._Loads
```

````

````{py:function} _compute_plan_cost(routes: src.policies.other.operators.repair.branch_bound._Plan, dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.other.operators.repair.branch_bound._compute_plan_cost

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._compute_plan_cost
```
````

````{py:function} _get_sorted_positions(node: int, routes: src.policies.other.operators.repair.branch_bound._Plan, loads: src.policies.other.operators.repair.branch_bound._Loads, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float) -> typing.List[src.policies.other.operators.repair.branch_bound._Position]
:canonical: src.policies.other.operators.repair.branch_bound._get_sorted_positions

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._get_sorted_positions
```
````

````{py:function} _get_sorted_positions_profit(node: int, routes: src.policies.other.operators.repair.branch_bound._Plan, loads: src.policies.other.operators.repair.branch_bound._Loads, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, is_mandatory: bool, seed_hurdle_factor: float = 0.5) -> typing.List[typing.Tuple[float, int, int]]
:canonical: src.policies.other.operators.repair.branch_bound._get_sorted_positions_profit

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._get_sorted_positions_profit
```
````

````{py:function} _lds_reinsert(routes: src.policies.other.operators.repair.branch_bound._Plan, loads: src.policies.other.operators.repair.branch_bound._Loads, unrouted: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, discrep: int, best_state: typing.List[typing.Optional[typing.Tuple[float, src.policies.other.operators.repair.branch_bound._Plan, src.policies.other.operators.repair.branch_bound._Loads]]]) -> None
:canonical: src.policies.other.operators.repair.branch_bound._lds_reinsert

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._lds_reinsert
```
````

````{py:function} _lds_reinsert_profit(routes: src.policies.other.operators.repair.branch_bound._Plan, loads: src.policies.other.operators.repair.branch_bound._Loads, unrouted: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_set: typing.Set[int], discrep: int, best_state: typing.List[typing.Optional[typing.Tuple[float, src.policies.other.operators.repair.branch_bound._Plan, src.policies.other.operators.repair.branch_bound._Loads]]], seed_hurdle_factor: float) -> None
:canonical: src.policies.other.operators.repair.branch_bound._lds_reinsert_profit

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound._lds_reinsert_profit
```
````

````{py:function} bb_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, max_discrepancy: int = 2, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.branch_bound.bb_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound.bb_insertion
```
````

````{py:function} bb_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, max_discrepancy: int = 2, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False, seed_hurdle_factor: float = 0.5) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.branch_bound.bb_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.branch_bound.bb_profit_insertion
```
````
