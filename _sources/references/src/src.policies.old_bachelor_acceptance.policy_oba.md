# {py:mod}`src.policies.old_bachelor_acceptance.policy_oba`

```{py:module} src.policies.old_bachelor_acceptance.policy_oba
```

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.policy_oba
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OBAPolicy <src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy>`
  - ```{autodoc2-docstring} src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy
    :summary:
    ```
````

### API

`````{py:class} OBAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.oba.OBAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.old_bachelor_acceptance.policy_oba.OBAPolicy._run_solver

````

`````
