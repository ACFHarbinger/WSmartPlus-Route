# {py:mod}`src.policies.particle_swarm_optimization.policy_pso`

```{py:module} src.policies.particle_swarm_optimization.policy_pso
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOPolicyAdapter <src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter
    :summary:
    ```
````

### API

`````{py:class} PSOPolicyAdapter(config: typing.Optional[typing.Union[logic.src.configs.policies.pso.PSOConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter._run_solver

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter._run_solver
```

````

````{py:method} get_name() -> str
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.get_name

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.get_name
```

````

````{py:method} get_acronym() -> str
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.get_acronym

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.get_acronym
```

````

`````
