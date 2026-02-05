# {py:mod}`src.policies.adapters.policy_hyper_aco`

```{py:module} src.policies.adapters.policy_hyper_aco
```

```{autodoc2-docstring} src.policies.adapters.policy_hyper_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperACOPolicy <src.policies.adapters.policy_hyper_aco.HyperACOPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_hyper_aco.HyperACOPolicy
    :summary:
    ```
````

### API

`````{py:class} HyperACOPolicy
:canonical: src.policies.adapters.policy_hyper_aco.HyperACOPolicy

Bases: {py:obj}`src.policies.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_hyper_aco.HyperACOPolicy
```

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_hyper_aco.HyperACOPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_hyper_aco.HyperACOPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.policy_hyper_aco.HyperACOPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_hyper_aco.HyperACOPolicy._run_solver
```

````

````{py:method} _build_greedy_solution(nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float) -> typing.List[typing.List[int]]
:canonical: src.policies.adapters.policy_hyper_aco.HyperACOPolicy._build_greedy_solution

```{autodoc2-docstring} src.policies.adapters.policy_hyper_aco.HyperACOPolicy._build_greedy_solution
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, cost_unit: float) -> float
:canonical: src.policies.adapters.policy_hyper_aco.HyperACOPolicy._calculate_cost

```{autodoc2-docstring} src.policies.adapters.policy_hyper_aco.HyperACOPolicy._calculate_cost
```

````

`````
