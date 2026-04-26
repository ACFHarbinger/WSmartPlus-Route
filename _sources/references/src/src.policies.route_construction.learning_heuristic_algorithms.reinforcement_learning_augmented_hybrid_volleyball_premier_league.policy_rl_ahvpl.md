# {py:mod}`src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl`

```{py:module} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLAHVPLPolicy <src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy
    :summary:
    ```
````

### API

`````{py:class} RLAHVPLPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.rl_ahvpl.RLAHVPLConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl.RLAHVPLPolicy._run_solver
```

````

`````
