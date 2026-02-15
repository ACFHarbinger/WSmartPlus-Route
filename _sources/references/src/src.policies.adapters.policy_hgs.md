# {py:mod}`src.policies.adapters.policy_hgs`

```{py:module} src.policies.adapters.policy_hgs
```

```{autodoc2-docstring} src.policies.adapters.policy_hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSPolicy <src.policies.adapters.policy_hgs.HGSPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_hgs.HGSPolicy
    :summary:
    ```
````

### API

`````{py:class} HGSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HGSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.adapters.policy_hgs.HGSPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_hgs.HGSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_hgs.HGSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.policy_hgs.HGSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_hgs.HGSPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_hgs.HGSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.policy_hgs.HGSPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_hgs.HGSPolicy._run_solver
```

````

`````
