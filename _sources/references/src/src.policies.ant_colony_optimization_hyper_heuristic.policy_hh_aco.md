# {py:mod}`src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco`

```{py:module} src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco
```

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperACOPolicy <src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy
    :summary:
    ```
````

### API

`````{py:class} HyperACOPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HyperHeuristicACOConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy._get_config_key

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy._run_solver

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco.HyperACOPolicy._run_solver
```

````

`````
