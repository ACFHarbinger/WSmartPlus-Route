# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProgressiveHedgingPolicy <src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy
    :summary:
    ```
````

### API

`````{py:class} ProgressiveHedgingPolicy(config: logic.src.configs.policies.PHConfig)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.PHConfig]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._get_config_key
```

````

````{py:method} _run_multi_period_solver(tree: typing.Any, capacity: float, revenue: float, cost_unit: float, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._run_multi_period_solver
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging.policy_ph.ProgressiveHedgingPolicy._run_solver
```

````

`````
