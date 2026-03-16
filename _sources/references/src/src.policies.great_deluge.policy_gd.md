# {py:mod}`src.policies.great_deluge.policy_gd`

```{py:module} src.policies.great_deluge.policy_gd
```

```{autodoc2-docstring} src.policies.great_deluge.policy_gd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GreatDelugePolicy <src.policies.great_deluge.policy_gd.GreatDelugePolicy>`
  - ```{autodoc2-docstring} src.policies.great_deluge.policy_gd.GreatDelugePolicy
    :summary:
    ```
````

### API

`````{py:class} GreatDelugePolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.gd.GDConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.great_deluge.policy_gd.GreatDelugePolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.great_deluge.policy_gd.GreatDelugePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.great_deluge.policy_gd.GreatDelugePolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.great_deluge.policy_gd.GreatDelugePolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.great_deluge.policy_gd.GreatDelugePolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.great_deluge.policy_gd.GreatDelugePolicy._run_solver

````

`````
