# {py:mod}`src.policies.helpers.operators.repair.forward_looking`

```{py:module} src.policies.helpers.operators.repair.forward_looking
```

```{autodoc2-docstring} src.policies.helpers.operators.repair.forward_looking
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compute_routing_delta <src.policies.helpers.operators.repair.forward_looking._compute_routing_delta>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.forward_looking._compute_routing_delta
    :summary:
    ```
* - {py:obj}`_simulate_inventory_forward <src.policies.helpers.operators.repair.forward_looking._simulate_inventory_forward>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.forward_looking._simulate_inventory_forward
    :summary:
    ```
* - {py:obj}`forward_looking_insertion <src.policies.helpers.operators.repair.forward_looking.forward_looking_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.forward_looking.forward_looking_insertion
    :summary:
    ```
````

### API

````{py:function} _compute_routing_delta(route: typing.List[int], node: int, position: int, dist_matrix: typing.Any, C: float) -> float
:canonical: src.policies.helpers.operators.repair.forward_looking._compute_routing_delta

```{autodoc2-docstring} src.policies.helpers.operators.repair.forward_looking._compute_routing_delta
```
````

````{py:function} _simulate_inventory_forward(node: int, visit_days: typing.List[int], initial_fill: float, demand_scenarios: typing.List[typing.List[float]], bin_capacity: float, t_start: int, H: int, stockout_penalty: float) -> float
:canonical: src.policies.helpers.operators.repair.forward_looking._simulate_inventory_forward

```{autodoc2-docstring} src.policies.helpers.operators.repair.forward_looking._simulate_inventory_forward
```
````

````{py:function} forward_looking_insertion(horizon_routes: typing.List[typing.List[typing.List[int]]], removed: typing.List[typing.Tuple[int, int]], dist_matrix: typing.Any, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, scenario_tree: typing.Optional[typing.Any] = None, stockout_penalty: float = 500.0, look_ahead_days: int = 3, lambda_inventory: float = 1.0) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.helpers.operators.repair.forward_looking.forward_looking_insertion

```{autodoc2-docstring} src.policies.helpers.operators.repair.forward_looking.forward_looking_insertion
```
````
