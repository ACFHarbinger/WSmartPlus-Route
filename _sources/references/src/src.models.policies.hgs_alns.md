# {py:mod}`src.models.policies.hgs_alns`

```{py:module} src.models.policies.hgs_alns
```

```{autodoc2-docstring} src.models.policies.hgs_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedHGSALNSEngine <src.models.policies.hgs_alns.VectorizedHGSALNSEngine>`
  - ```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNSEngine
    :summary:
    ```
* - {py:obj}`VectorizedHGSALNS <src.models.policies.hgs_alns.VectorizedHGSALNS>`
  - ```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS
    :summary:
    ```
````

### API

`````{py:class} VectorizedHGSALNSEngine(dist_matrix: torch.Tensor, wastes: torch.Tensor, vehicle_capacity: typing.Any, time_limit: float = 1.0, device: str = 'cuda', alns_education_iterations: int = 50)
:canonical: src.models.policies.hgs_alns.VectorizedHGSALNSEngine

Bases: {py:obj}`logic.src.models.policies.hybrid_genetic_search.VectorizedHGS`

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNSEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNSEngine.__init__
```

````{py:method} educate(routes_list: list[list[int]], split_costs: torch.Tensor, max_vehicles: int = 0) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.hgs_alns.VectorizedHGSALNSEngine.educate

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNSEngine.educate
```

````

`````

`````{py:class} VectorizedHGSALNS(env_name: str, time_limit: float = 5.0, population_size: int = 20, n_generations: int = 10, elite_size: int = 5, max_vehicles: int = 0, alns_education_iterations: int = 50, **kwargs)
:canonical: src.models.policies.hgs_alns.VectorizedHGSALNS

Bases: {py:obj}`logic.src.models.policies.hgs.VectorizedHGS`

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.hgs_alns.VectorizedHGSALNS.forward

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS.forward
```

````

`````
