# {py:mod}`src.policies.helpers.operators.recreate_repair.multi_period`

```{py:module} src.policies.helpers.operators.recreate_repair.multi_period
```

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`greedy_horizon_insertion <src.policies.helpers.operators.recreate_repair.multi_period.greedy_horizon_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period.greedy_horizon_insertion
    :summary:
    ```
* - {py:obj}`regret_k_temporal_insertion <src.policies.helpers.operators.recreate_repair.multi_period.regret_k_temporal_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period.regret_k_temporal_insertion
    :summary:
    ```
* - {py:obj}`stochastic_aware_insertion <src.policies.helpers.operators.recreate_repair.multi_period.stochastic_aware_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period.stochastic_aware_insertion
    :summary:
    ```
* - {py:obj}`_get_scenarios <src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios
    :summary:
    ```
* - {py:obj}`_get_scenarios_from_ef <src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_ef>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_ef
    :summary:
    ```
* - {py:obj}`_get_scenarios_from_prediction <src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_prediction>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_prediction
    :summary:
    ```
````

### API

````{py:function} greedy_horizon_insertion(horizon_routes: typing.List[typing.List[typing.List[int]]], removed: typing.List[typing.Tuple[int, int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, scenario_tree: typing.Optional[typing.Any] = None, use_stochastic: bool = False, stockout_penalty: float = 500.0, look_ahead_days: int = 3) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.helpers.operators.recreate_repair.multi_period.greedy_horizon_insertion

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period.greedy_horizon_insertion
```
````

````{py:function} regret_k_temporal_insertion(horizon_routes: typing.List[typing.List[typing.List[int]]], removed: typing.List[typing.Tuple[int, int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, k: int = 2, scenario_tree: typing.Optional[typing.Any] = None, use_stochastic: bool = False, stockout_penalty: float = 500.0) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.helpers.operators.recreate_repair.multi_period.regret_k_temporal_insertion

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period.regret_k_temporal_insertion
```
````

````{py:function} stochastic_aware_insertion(horizon_routes: typing.List[typing.List[typing.List[int]]], removed: typing.List[typing.Tuple[int, int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, scenario_tree: typing.Optional[typing.Any] = None, stockout_penalty: float = 500.0) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.helpers.operators.recreate_repair.multi_period.stochastic_aware_insertion

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period.stochastic_aware_insertion
```
````

````{py:function} _get_scenarios(tree: typing.Any, node: int, t: int, H: int) -> typing.List[typing.List[float]]
:canonical: src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios
```
````

````{py:function} _get_scenarios_from_ef(tree: typing.Any, node: int, t: int, H: int) -> typing.List[typing.List[float]]
:canonical: src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_ef

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_ef
```
````

````{py:function} _get_scenarios_from_prediction(tree: typing.Any, node: int, t: int, H: int) -> typing.List[typing.List[float]]
:canonical: src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_prediction

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.multi_period._get_scenarios_from_prediction
```
````
