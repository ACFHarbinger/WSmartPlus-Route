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

`````{py:class} PSOPolicyAdapter(**config: typing.Any)
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter

Bases: {py:obj}`logic.src.interfaces.adapter.IPolicyAdapter`

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.__init__
```

````{py:method} __call__(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.__call__

```{autodoc2-docstring} src.policies.particle_swarm_optimization.policy_pso.PSOPolicyAdapter.__call__
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
