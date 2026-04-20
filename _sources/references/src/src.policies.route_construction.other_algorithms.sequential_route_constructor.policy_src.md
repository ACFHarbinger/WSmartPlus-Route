# {py:mod}`src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src`

```{py:module} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SequentialRouteConstructor <src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.logger
```

````

`````{py:class} SequentialRouteConstructor(config: typing.Any = None)
:canonical: src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor.__init__
```

````{py:method} _config_class()
:canonical: src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor._get_config_key
:classmethod:

````

````{py:method} _initialize_constructors() -> None
:canonical: src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor._initialize_constructors

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor._initialize_constructors
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.Union[typing.List[int], typing.List[typing.List[int]]], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor.execute

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor.execute
```

````

````{py:method} _run_solver(sub_dist_matrix: typing.Any, sub_wastes: typing.Any, capacity: float, revenue: float, cost_unit: float, values: typing.Any, mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor._run_solver

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.sequential_route_constructor.policy_src.SequentialRouteConstructor._run_solver
```

````

`````
