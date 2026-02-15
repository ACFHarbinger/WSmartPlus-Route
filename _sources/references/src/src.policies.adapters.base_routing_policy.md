# {py:mod}`src.policies.adapters.base_routing_policy`

```{py:module} src.policies.adapters.base_routing_policy
```

```{autodoc2-docstring} src.policies.adapters.base_routing_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseRoutingPolicy <src.policies.adapters.base_routing_policy.BaseRoutingPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_flatten_raw_config <src.policies.adapters.base_routing_policy._flatten_raw_config>`
  - ```{autodoc2-docstring} src.policies.adapters.base_routing_policy._flatten_raw_config
    :summary:
    ```
````

### API

````{py:function} _flatten_raw_config(source: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.policies.adapters.base_routing_policy._flatten_raw_config

```{autodoc2-docstring} src.policies.adapters.base_routing_policy._flatten_raw_config
```
````

`````{py:class} BaseRoutingPolicy(config: typing.Any = None)
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy

Bases: {py:obj}`logic.src.interfaces.adapter.IPolicyAdapter`

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy.__init__
```

````{py:property} config
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy.config
:type: typing.Any

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy.config
```

````

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._config_class
```

````

````{py:method} _build_config(raw_config: typing.Dict[str, typing.Any]) -> typing.Any
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._build_config
:classmethod:

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._build_config
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._get_config_key
```

````

````{py:method} _validate_must_go(must_go: typing.Optional[typing.List[int]]) -> typing.Optional[typing.Tuple[typing.List[int], float, typing.Any]]
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._validate_must_go

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._validate_must_go
```

````

````{py:method} _load_area_params(area: str, waste_type: str, config: typing.Dict[str, typing.Any]) -> typing.Tuple[float, float, float, typing.Dict[str, typing.Any]]
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._load_area_params

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._load_area_params
```

````

````{py:method} _create_subset_problem(must_go: typing.List[int], distance_matrix: typing.Any, bins: typing.Any) -> typing.Tuple[numpy.ndarray, typing.Dict[int, float], typing.List[int]]
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._create_subset_problem

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._create_subset_problem
```

````

````{py:method} _map_tour_to_global(routes: typing.List[typing.List[int]], subset_indices: typing.List[int]) -> typing.List[int]
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._map_tour_to_global

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._map_tour_to_global
```

````

````{py:method} _compute_cost(distance_matrix: typing.Any, tour: typing.List[int]) -> float
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._compute_cost

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._compute_cost
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy._run_solver
:abstractmethod:

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.adapters.base_routing_policy.BaseRoutingPolicy.execute

```{autodoc2-docstring} src.policies.adapters.base_routing_policy.BaseRoutingPolicy.execute
```

````

`````
