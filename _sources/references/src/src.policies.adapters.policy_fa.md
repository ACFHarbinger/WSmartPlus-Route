# {py:mod}`src.policies.adapters.policy_fa`

```{py:module} src.policies.adapters.policy_fa
```

```{autodoc2-docstring} src.policies.adapters.policy_fa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FAPolicy <src.policies.adapters.policy_fa.FAPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_fa.FAPolicy
    :summary:
    ```
````

### API

`````{py:class} FAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.fa.FAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.adapters.policy_fa.FAPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_fa.FAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_fa.FAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.policy_fa.FAPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_fa.FAPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adapters.policy_fa.FAPolicy._run_solver

````

`````
