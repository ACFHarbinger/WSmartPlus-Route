# {py:mod}`src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma`

```{py:module} src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOMAPolicy <src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy>`
  - ```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy
    :summary:
    ```
````

### API

`````{py:class} PSOMAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.psoma.PSOMAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma.PSOMAPolicy._run_solver

````

`````
