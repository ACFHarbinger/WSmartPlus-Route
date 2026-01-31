# {py:mod}`src.policies.adapters.policy_hgs_alns`

```{py:module} src.policies.adapters.policy_hgs_alns
```

```{autodoc2-docstring} src.policies.adapters.policy_hgs_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSALNSPolicy <src.policies.adapters.policy_hgs_alns.HGSALNSPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_hgs_alns.HGSALNSPolicy
    :summary:
    ```
````

### API

`````{py:class} HGSALNSPolicy
:canonical: src.policies.adapters.policy_hgs_alns.HGSALNSPolicy

Bases: {py:obj}`src.policies.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_hgs_alns.HGSALNSPolicy
```

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_hgs_alns.HGSALNSPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_hgs_alns.HGSALNSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.policy_hgs_alns.HGSALNSPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_hgs_alns.HGSALNSPolicy._run_solver
```

````

`````
