# {py:mod}`src.policies.adapters.policy_hils`

```{py:module} src.policies.adapters.policy_hils
```

```{autodoc2-docstring} src.policies.adapters.policy_hils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HILSPolicy <src.policies.adapters.policy_hils.HILSPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_hils.HILSPolicy
    :summary:
    ```
````

### API

`````{py:class} HILSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HILSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.adapters.policy_hils.HILSPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_hils.HILSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_hils.HILSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.policy_hils.HILSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_hils.HILSPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_hils.HILSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adapters.policy_hils.HILSPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_hils.HILSPolicy._run_solver
```

````

`````
