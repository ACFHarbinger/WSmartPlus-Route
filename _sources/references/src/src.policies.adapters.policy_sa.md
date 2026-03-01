# {py:mod}`src.policies.adapters.policy_sa`

```{py:module} src.policies.adapters.policy_sa
```

```{autodoc2-docstring} src.policies.adapters.policy_sa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAPolicy <src.policies.adapters.policy_sa.SAPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_sa.SAPolicy
    :summary:
    ```
````

### API

`````{py:class} SAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.sa.SAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.adapters.policy_sa.SAPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_sa.SAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_sa.SAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.policy_sa.SAPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_sa.SAPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adapters.policy_sa.SAPolicy._run_solver

````

`````
