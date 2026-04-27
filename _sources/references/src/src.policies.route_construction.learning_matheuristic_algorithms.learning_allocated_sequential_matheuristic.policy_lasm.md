# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LASMPolicy <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy
    :summary:
    ```
````

### API

`````{py:class} LASMPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.LASMPipelineConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.policy_lasm.LASMPolicy._run_solver
```

````

`````
