# {py:mod}`src.policies.route_improvement.set_covering`

```{py:module} src.policies.route_improvement.set_covering
```

```{autodoc2-docstring} src.policies.route_improvement.set_covering
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SetCoverRouteImprover <src.policies.route_improvement.set_covering.SetCoverRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_covering.SetCoverRouteImprover
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_canonical <src.policies.route_improvement.set_covering._canonical>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_covering._canonical
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.set_covering.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_covering.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.set_covering.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.set_covering.logger
```

````

````{py:function} _canonical(route: typing.List[int]) -> typing.Tuple[int, ...]
:canonical: src.policies.route_improvement.set_covering._canonical

```{autodoc2-docstring} src.policies.route_improvement.set_covering._canonical
```
````

`````{py:class} SetCoverRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.set_covering.SetCoverRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.set_covering.SetCoverRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.set_covering.SetCoverRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.set_covering.SetCoverRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.set_covering.SetCoverRouteImprover.process
```

````

````{py:method} _solve_set_cover_ip(pool: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: dict, cost_per_km: float, revenue_kg: float, target_nodes: typing.Set[int], time_limit: float) -> typing.List[typing.List[int]]
:canonical: src.policies.route_improvement.set_covering.SetCoverRouteImprover._solve_set_cover_ip

```{autodoc2-docstring} src.policies.route_improvement.set_covering.SetCoverRouteImprover._solve_set_cover_ip
```

````

````{py:method} _deduplicate_nodes(routes: typing.List[typing.List[int]], dm: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.route_improvement.set_covering.SetCoverRouteImprover._deduplicate_nodes

```{autodoc2-docstring} src.policies.route_improvement.set_covering.SetCoverRouteImprover._deduplicate_nodes
```

````

`````
