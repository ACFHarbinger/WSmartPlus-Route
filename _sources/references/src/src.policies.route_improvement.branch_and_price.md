# {py:mod}`src.policies.route_improvement.branch_and_price`

```{py:module} src.policies.route_improvement.branch_and_price
```

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndPriceRouteImprover <src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_time_limit <src.policies.route_improvement.branch_and_price._time_limit>`
  - ```{autodoc2-docstring} src.policies.route_improvement.branch_and_price._time_limit
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.branch_and_price.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.branch_and_price.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.logger
```

````

````{py:function} _time_limit(seconds: float)
:canonical: src.policies.route_improvement.branch_and_price._time_limit

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price._time_limit
```
````

`````{py:class} BranchAndPriceRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover.process
```

````

````{py:method} _solve_inhouse(input_routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, cost_per_km: float, revenue_kg: float, mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Optional[typing.List[typing.List[int]]]
:canonical: src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover._solve_inhouse

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover._solve_inhouse
```

````

````{py:method} _solve_vrpy(input_routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, cost_per_km: float, revenue_kg: float, mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Optional[typing.List[typing.List[int]]]
:canonical: src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover._solve_vrpy

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover._solve_vrpy
```

````

````{py:method} _fallback_set_partitioning(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover._fallback_set_partitioning

```{autodoc2-docstring} src.policies.route_improvement.branch_and_price.BranchAndPriceRouteImprover._fallback_set_partitioning
```

````

`````
