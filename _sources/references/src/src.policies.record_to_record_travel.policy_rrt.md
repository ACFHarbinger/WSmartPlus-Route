# {py:mod}`src.policies.record_to_record_travel.policy_rrt`

```{py:module} src.policies.record_to_record_travel.policy_rrt
```

```{autodoc2-docstring} src.policies.record_to_record_travel.policy_rrt
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RRTPolicy <src.policies.record_to_record_travel.policy_rrt.RRTPolicy>`
  - ```{autodoc2-docstring} src.policies.record_to_record_travel.policy_rrt.RRTPolicy
    :summary:
    ```
````

### API

`````{py:class} RRTPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.rrt.RRTConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.record_to_record_travel.policy_rrt.RRTPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.record_to_record_travel.policy_rrt.RRTPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.record_to_record_travel.policy_rrt.RRTPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.record_to_record_travel.policy_rrt.RRTPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.record_to_record_travel.policy_rrt.RRTPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.record_to_record_travel.policy_rrt.RRTPolicy._run_solver

````

`````
