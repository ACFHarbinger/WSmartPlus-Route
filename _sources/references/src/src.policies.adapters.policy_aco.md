# {py:mod}`src.policies.adapters.policy_aco`

```{py:module} src.policies.adapters.policy_aco
```

```{autodoc2-docstring} src.policies.adapters.policy_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACOPolicy <src.policies.adapters.policy_aco.ACOPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_aco.ACOPolicy
    :summary:
    ```
````

### API

`````{py:class} ACOPolicy
:canonical: src.policies.adapters.policy_aco.ACOPolicy

Bases: {py:obj}`src.policies.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_aco.ACOPolicy
```

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_aco.ACOPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_aco.ACOPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.policy_aco.ACOPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_aco.ACOPolicy._run_solver
```

````

`````
