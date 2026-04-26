# {py:mod}`src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh`

```{py:module} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperACOPolicy <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy
    :summary:
    ```
````

### API

`````{py:class} HyperACOPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HyperHeuristicACOConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh.HyperACOPolicy._run_solver
```

````

`````
