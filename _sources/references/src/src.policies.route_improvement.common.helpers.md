# {py:mod}`src.policies.route_improvement.common.helpers`

```{py:module} src.policies.route_improvement.common.helpers
```

```{autodoc2-docstring} src.policies.route_improvement.common.helpers
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`resolve_mandatory_nodes <src.policies.route_improvement.common.helpers.resolve_mandatory_nodes>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.resolve_mandatory_nodes
    :summary:
    ```
* - {py:obj}`upgrade_repair_op_to_profit <src.policies.route_improvement.common.helpers.upgrade_repair_op_to_profit>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.upgrade_repair_op_to_profit
    :summary:
    ```
* - {py:obj}`to_numpy <src.policies.route_improvement.common.helpers.to_numpy>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.to_numpy
    :summary:
    ```
* - {py:obj}`split_tour <src.policies.route_improvement.common.helpers.split_tour>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.split_tour
    :summary:
    ```
* - {py:obj}`assemble_tour <src.policies.route_improvement.common.helpers.assemble_tour>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.assemble_tour
    :summary:
    ```
* - {py:obj}`route_distance <src.policies.route_improvement.common.helpers.route_distance>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.route_distance
    :summary:
    ```
* - {py:obj}`tour_distance <src.policies.route_improvement.common.helpers.tour_distance>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.tour_distance
    :summary:
    ```
* - {py:obj}`route_load <src.policies.route_improvement.common.helpers.route_load>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.helpers.route_load
    :summary:
    ```
````

### API

````{py:function} resolve_mandatory_nodes(kwargs: typing.Dict[str, typing.Any], config: typing.Dict[str, typing.Any]) -> typing.Optional[typing.List[int]]
:canonical: src.policies.route_improvement.common.helpers.resolve_mandatory_nodes

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.resolve_mandatory_nodes
```
````

````{py:function} upgrade_repair_op_to_profit(repair_op: str, revenue_kg: float, cost_per_km: float) -> str
:canonical: src.policies.route_improvement.common.helpers.upgrade_repair_op_to_profit

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.upgrade_repair_op_to_profit
```
````

````{py:function} to_numpy(distance_matrix: typing.Any) -> numpy.ndarray
:canonical: src.policies.route_improvement.common.helpers.to_numpy

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.to_numpy
```
````

````{py:function} split_tour(tour: typing.Sequence[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_improvement.common.helpers.split_tour

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.split_tour
```
````

````{py:function} assemble_tour(routes: typing.Sequence[typing.Sequence[int]]) -> typing.List[int]
:canonical: src.policies.route_improvement.common.helpers.assemble_tour

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.assemble_tour
```
````

````{py:function} route_distance(route: typing.Sequence[int], dm: numpy.ndarray) -> float
:canonical: src.policies.route_improvement.common.helpers.route_distance

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.route_distance
```
````

````{py:function} tour_distance(routes: typing.Iterable[typing.Sequence[int]], dm: numpy.ndarray) -> float
:canonical: src.policies.route_improvement.common.helpers.tour_distance

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.tour_distance
```
````

````{py:function} route_load(route: typing.Sequence[int], wastes: typing.Dict[int, float]) -> float
:canonical: src.policies.route_improvement.common.helpers.route_load

```{autodoc2-docstring} src.policies.route_improvement.common.helpers.route_load
```
````
