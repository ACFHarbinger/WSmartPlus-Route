# {py:mod}`src.utils.policy.routes`

```{py:module} src.utils.policy.routes
```

```{autodoc2-docstring} src.utils.policy.routes
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`prune_unprofitable_routes <src.utils.policy.routes.prune_unprofitable_routes>`
  - ```{autodoc2-docstring} src.utils.policy.routes.prune_unprofitable_routes
    :summary:
    ```
* - {py:obj}`route_profit <src.utils.policy.routes.route_profit>`
  - ```{autodoc2-docstring} src.utils.policy.routes.route_profit
    :summary:
    ```
* - {py:obj}`route_cost <src.utils.policy.routes.route_cost>`
  - ```{autodoc2-docstring} src.utils.policy.routes.route_cost
    :summary:
    ```
````

### API

````{py:function} prune_unprofitable_routes(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, mandatory_nodes_set: typing.Set[int]) -> typing.List[typing.List[int]]
:canonical: src.utils.policy.routes.prune_unprofitable_routes

```{autodoc2-docstring} src.utils.policy.routes.prune_unprofitable_routes
```
````

````{py:function} route_profit(route: typing.List[int], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.utils.policy.routes.route_profit

```{autodoc2-docstring} src.utils.policy.routes.route_profit
```
````

````{py:function} route_cost(route: typing.List[int], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.utils.policy.routes.route_cost

```{autodoc2-docstring} src.utils.policy.routes.route_cost
```
````
