# {py:mod}`src.policies.step_counting_hill_climbing.policy_schc`

```{py:module} src.policies.step_counting_hill_climbing.policy_schc
```

```{autodoc2-docstring} src.policies.step_counting_hill_climbing.policy_schc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepCountingHillClimbingPolicy <src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy>`
  - ```{autodoc2-docstring} src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy
    :summary:
    ```
````

### API

`````{py:class} StepCountingHillClimbingPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.schc.SCHCConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.step_counting_hill_climbing.policy_schc.StepCountingHillClimbingPolicy._run_solver

````

`````
