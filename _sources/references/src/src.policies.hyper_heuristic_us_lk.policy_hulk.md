# {py:mod}`src.policies.hyper_heuristic_us_lk.policy_hulk`

```{py:module} src.policies.hyper_heuristic_us_lk.policy_hulk
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.policy_hulk
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HULKPolicy <src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy>`
  - ```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy
    :summary:
    ```
````

### API

`````{py:class} HULKPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.hulk.HULKConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hyper_heuristic_us_lk.policy_hulk.HULKPolicy._run_solver

````

`````
