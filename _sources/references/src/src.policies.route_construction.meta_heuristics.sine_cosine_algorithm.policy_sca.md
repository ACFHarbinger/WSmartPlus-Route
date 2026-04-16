# {py:mod}`src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca`

```{py:module} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCAPolicy <src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy
    :summary:
    ```
````

### API

`````{py:class} SCAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.sca.SCAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.policy_sca.SCAPolicy._run_solver
```

````

`````
