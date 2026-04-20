# {py:mod}`src.policies.helpers.operators.intensification_fixing.set_partitioning_polish`

```{py:module} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish
```

```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_route_cost <src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_cost
    :summary:
    ```
* - {py:obj}`_route_profit <src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_profit
    :summary:
    ```
* - {py:obj}`_deduplicate_pool <src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._deduplicate_pool>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._deduplicate_pool
    :summary:
    ```
* - {py:obj}`set_partitioning_polish <src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish
    :summary:
    ```
* - {py:obj}`set_partitioning_polish_profit <src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish_profit
    :summary:
    ```
````

### API

````{py:function} _route_cost(route: typing.List[int], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_cost

```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_cost
```
````

````{py:function} _route_profit(route: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float) -> float
:canonical: src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_profit

```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._route_profit
```
````

````{py:function} _deduplicate_pool(route_pool: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._deduplicate_pool

```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish._deduplicate_pool
```
````

````{py:function} set_partitioning_polish(routes: typing.List[typing.List[int]], route_pool: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, time_limit: float = 60.0, seed: int = 42) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish

```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish
```
````

````{py:function} set_partitioning_polish_profit(routes: typing.List[typing.List[int]], route_pool: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, time_limit: float = 60.0, seed: int = 42) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish_profit

```{autodoc2-docstring} src.policies.helpers.operators.intensification_fixing.set_partitioning_polish.set_partitioning_polish_profit
```
````
