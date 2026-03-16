# {py:mod}`src.policies.threshold_accepting.policy_ta`

```{py:module} src.policies.threshold_accepting.policy_ta
```

```{autodoc2-docstring} src.policies.threshold_accepting.policy_ta
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThresholdAcceptingPolicy <src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy>`
  - ```{autodoc2-docstring} src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy
    :summary:
    ```
````

### API

`````{py:class} ThresholdAcceptingPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ta.TAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.threshold_accepting.policy_ta.ThresholdAcceptingPolicy._run_solver

````

`````
