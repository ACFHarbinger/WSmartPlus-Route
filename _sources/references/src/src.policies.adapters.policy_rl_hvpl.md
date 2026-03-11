# {py:mod}`src.policies.adapters.policy_rl_hvpl`

```{py:module} src.policies.adapters.policy_rl_hvpl
```

```{autodoc2-docstring} src.policies.adapters.policy_rl_hvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLHVPLPolicy <src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy
    :summary:
    ```
````

### API

`````{py:class} RLHVPLPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.rl_hvpl.RLHVPLConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_rl_hvpl.RLHVPLPolicy._run_solver
```

````

`````
