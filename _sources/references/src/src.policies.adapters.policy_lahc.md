# {py:mod}`src.policies.adapters.policy_lahc`

```{py:module} src.policies.adapters.policy_lahc
```

```{autodoc2-docstring} src.policies.adapters.policy_lahc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LAHCPolicy <src.policies.adapters.policy_lahc.LAHCPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_lahc.LAHCPolicy
    :summary:
    ```
````

### API

`````{py:class} LAHCPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.lahc.LAHCConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.adapters.policy_lahc.LAHCPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_lahc.LAHCPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_lahc.LAHCPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.policy_lahc.LAHCPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_lahc.LAHCPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adapters.policy_lahc.LAHCPolicy._run_solver

````

`````
