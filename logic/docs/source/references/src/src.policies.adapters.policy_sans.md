# {py:mod}`src.policies.adapters.policy_sans`

```{py:module} src.policies.adapters.policy_sans
```

```{autodoc2-docstring} src.policies.adapters.policy_sans
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SANSPolicy <src.policies.adapters.policy_sans.SANSPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_sans.SANSPolicy
    :summary:
    ```
````

### API

`````{py:class} SANSPolicy
:canonical: src.policies.adapters.policy_sans.SANSPolicy

Bases: {py:obj}`src.policies.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_sans.SANSPolicy
```

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_sans.SANSPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_sans.SANSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.policy_sans.SANSPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_sans.SANSPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.adapters.policy_sans.SANSPolicy.execute

```{autodoc2-docstring} src.policies.adapters.policy_sans.SANSPolicy.execute
```

````

`````
