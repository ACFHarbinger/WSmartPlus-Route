# {py:mod}`src.models.policies.hgs`

```{py:module} src.models.policies.hgs
```

```{autodoc2-docstring} src.models.policies.hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedHGS <src.models.policies.hgs.VectorizedHGS>`
  - ```{autodoc2-docstring} src.models.policies.hgs.VectorizedHGS
    :summary:
    ```
````

### API

`````{py:class} VectorizedHGS(env_name: str, time_limit: float = 5.0, population_size: int = 50, n_generations: int = 50, elite_size: int = 10, max_vehicles: int = 0, crossover_rate: float = 0.7, max_iterations: int = 50, seed: int = 42, device: str = 'cpu', **kwargs)
:canonical: src.models.policies.hgs.VectorizedHGS

Bases: {py:obj}`logic.src.models.common.autoregressive.policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.policies.hgs.VectorizedHGS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hgs.VectorizedHGS.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.hgs.VectorizedHGS.forward

```{autodoc2-docstring} src.models.policies.hgs.VectorizedHGS.forward
```

````

````{py:method} _compute_reward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase], actions: torch.Tensor) -> torch.Tensor
:canonical: src.models.policies.hgs.VectorizedHGS._compute_reward
:staticmethod:

```{autodoc2-docstring} src.models.policies.hgs.VectorizedHGS._compute_reward
```

````

`````
