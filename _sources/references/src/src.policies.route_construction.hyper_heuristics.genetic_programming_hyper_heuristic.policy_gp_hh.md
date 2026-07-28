# {py:mod}`src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh`

```{py:module} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPHHPolicy <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_make_synthetic_training_envs <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh._make_synthetic_training_envs>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh._make_synthetic_training_envs
    :summary:
    ```
````

### API

````{py:function} _make_synthetic_training_envs(n_nodes: int, n_envs: int, capacity: float, R: float, C: float, rng: numpy.random.Generator) -> typing.List[typing.Tuple[numpy.ndarray, typing.Dict[int, float], typing.List[int]]]
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh._make_synthetic_training_envs

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh._make_synthetic_training_envs
```
````

`````{py:class} GPHHPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.gp_hh.GPHHConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh.GPHHPolicy._run_solver
```

````

`````
