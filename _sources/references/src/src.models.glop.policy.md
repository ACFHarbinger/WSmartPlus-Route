# {py:mod}`src.models.glop.policy`

```{py:module} src.models.glop.policy
```

```{autodoc2-docstring} src.models.glop.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GLOPPolicy <src.models.glop.policy.GLOPPolicy>`
  - ```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SubProblemSolverType <src.models.glop.policy.SubProblemSolverType>`
  - ```{autodoc2-docstring} src.models.glop.policy.SubProblemSolverType
    :summary:
    ```
````

### API

````{py:data} SubProblemSolverType
:canonical: src.models.glop.policy.SubProblemSolverType
:value: >
   None

```{autodoc2-docstring} src.models.glop.policy.SubProblemSolverType
```

````

`````{py:class} GLOPPolicy(env_name: str = 'cvrp', n_samples: int = 10, temperature: float = 1.0, embed_dim: int = 64, subprob_solver: src.models.glop.policy.SubProblemSolverType | str = 'greedy', subprob_batch_size: int = 2000, **encoder_kwargs)
:canonical: src.models.glop.policy.GLOPPolicy

Bases: {py:obj}`logic.src.models.common.nonautoregressive_policy.NonAutoregressivePolicy`

```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, phase: typing.Literal[train, val, test] = 'test', calc_reward: bool = True, return_actions: bool = False, return_entropy: bool = False, strategy: typing.Optional[str] = None, **decoding_kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.glop.policy.GLOPPolicy.forward

```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy.forward
```

````

````{py:method} _local_policy(td: tensordict.TensorDict, partition_actions: torch.Tensor) -> typing.Dict[str, typing.Any]
:canonical: src.models.glop.policy.GLOPPolicy._local_policy

```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy._local_policy
```

````

````{py:method} _get_solver() -> src.models.glop.policy.SubProblemSolverType
:canonical: src.models.glop.policy.GLOPPolicy._get_solver

```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy._get_solver
```

````

````{py:method} _greedy_tsp_solver(coords: torch.Tensor) -> torch.Tensor
:canonical: src.models.glop.policy.GLOPPolicy._greedy_tsp_solver
:staticmethod:

```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy._greedy_tsp_solver
```

````

````{py:method} _nearest_neighbor_solver(coords: torch.Tensor) -> torch.Tensor
:canonical: src.models.glop.policy.GLOPPolicy._nearest_neighbor_solver
:staticmethod:

```{autodoc2-docstring} src.models.glop.policy.GLOPPolicy._nearest_neighbor_solver
```

````

`````
