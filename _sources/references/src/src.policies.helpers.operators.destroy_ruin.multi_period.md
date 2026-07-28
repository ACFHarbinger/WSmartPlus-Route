# {py:mod}`src.policies.helpers.operators.destroy_ruin.multi_period`

```{py:module} src.policies.helpers.operators.destroy_ruin.multi_period
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`random_horizon_removal <src.policies.helpers.operators.destroy_ruin.multi_period.random_horizon_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.random_horizon_removal
    :summary:
    ```
* - {py:obj}`worst_profit_horizon_removal <src.policies.helpers.operators.destroy_ruin.multi_period.worst_profit_horizon_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.worst_profit_horizon_removal
    :summary:
    ```
* - {py:obj}`shaw_horizon_removal <src.policies.helpers.operators.destroy_ruin.multi_period.shaw_horizon_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.shaw_horizon_removal
    :summary:
    ```
* - {py:obj}`urgency_aware_removal <src.policies.helpers.operators.destroy_ruin.multi_period.urgency_aware_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.urgency_aware_removal
    :summary:
    ```
* - {py:obj}`shift_visit_removal <src.policies.helpers.operators.destroy_ruin.multi_period.shift_visit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.shift_visit_removal
    :summary:
    ```
* - {py:obj}`pattern_removal <src.policies.helpers.operators.destroy_ruin.multi_period.pattern_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.pattern_removal
    :summary:
    ```
````

### API

````{py:function} random_horizon_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy_ruin.multi_period.random_horizon_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.random_horizon_removal
```
````

````{py:function} worst_profit_horizon_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, p: float = 3.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy_ruin.multi_period.worst_profit_horizon_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.worst_profit_horizon_removal
```
````

````{py:function} shaw_horizon_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, dist_matrix: numpy.ndarray, p: float = 6.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy_ruin.multi_period.shaw_horizon_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.shaw_horizon_removal
```
````

````{py:function} urgency_aware_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, wastes: typing.Dict[int, float], fill_threshold: float = 70.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy_ruin.multi_period.urgency_aware_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.urgency_aware_removal
```
````

````{py:function} shift_visit_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, direction: str = 'both', wastes: typing.Optional[typing.Dict[int, float]] = None, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy_ruin.multi_period.shift_visit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.shift_visit_removal
```
````

````{py:function} pattern_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, wastes: typing.Optional[typing.Dict[int, float]] = None, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy_ruin.multi_period.pattern_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.multi_period.pattern_removal
```
````
