# {py:mod}`src.policies.memetic_algorithm_dual_population.policy_ma_dp`

```{py:module} src.policies.memetic_algorithm_dual_population.policy_ma_dp
```

```{autodoc2-docstring} src.policies.memetic_algorithm_dual_population.policy_ma_dp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmDualPopulationPolicy <src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy>`
  - ```{autodoc2-docstring} src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmDualPopulationPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.memetic_algorithm_dual_population.policy_ma_dp.MemeticAlgorithmDualPopulationPolicy._run_solver

````

`````
