# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExactSDPPolicy <src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SDP_CACHE <src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE
    :summary:
    ```
````

### API

````{py:data} _SDP_CACHE
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE
:type: typing.Dict[typing.Tuple[int, int, int, float], logic.src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE
```

````

`````{py:class} ExactSDPPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.esdp.ExactSDPConfig]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._get_config_key
```

````

````{py:method} _run_multi_period_solver(tree: typing.Any, capacity: float, revenue: float, cost_unit: float, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._run_multi_period_solver
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._run_solver
```

````

`````
