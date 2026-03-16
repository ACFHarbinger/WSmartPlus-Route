# {py:mod}`src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts`

```{py:module} src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts
```

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmToleranceBasedSelectionPolicy <src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy>`
  - ```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmToleranceBasedSelectionPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.memetic_algorithm_tolerance_selection.policy_ma_ts.MemeticAlgorithmToleranceBasedSelectionPolicy._run_solver

````

`````
