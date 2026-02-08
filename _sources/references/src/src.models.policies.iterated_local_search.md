# {py:mod}`src.models.policies.iterated_local_search`

```{py:module} src.models.policies.iterated_local_search
```

```{autodoc2-docstring} src.models.policies.iterated_local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IteratedLocalSearchPolicy <src.models.policies.iterated_local_search.IteratedLocalSearchPolicy>`
  - ```{autodoc2-docstring} src.models.policies.iterated_local_search.IteratedLocalSearchPolicy
    :summary:
    ```
````

### API

`````{py:class} IteratedLocalSearchPolicy(env_name: str, ls_operator: typing.Union[str, dict[str, float]] = 'two_opt', perturbation_type: typing.Union[str, dict[str, float]] = 'double_bridge', n_restarts: int = 5, ls_iterations: int = 50, perturbation_strength: float = 0.2, **kwargs)
:canonical: src.models.policies.iterated_local_search.IteratedLocalSearchPolicy

Bases: {py:obj}`logic.src.models.common.improvement_policy.ImprovementPolicy`

```{autodoc2-docstring} src.models.policies.iterated_local_search.IteratedLocalSearchPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.iterated_local_search.IteratedLocalSearchPolicy.__init__
```

````{py:method} _perturb(tours: torch.Tensor, device: torch.device, mode: str) -> torch.Tensor
:canonical: src.models.policies.iterated_local_search.IteratedLocalSearchPolicy._perturb

```{autodoc2-docstring} src.models.policies.iterated_local_search.IteratedLocalSearchPolicy._perturb
```

````

````{py:method} _compute_costs(tours: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor
:canonical: src.models.policies.iterated_local_search.IteratedLocalSearchPolicy._compute_costs

```{autodoc2-docstring} src.models.policies.iterated_local_search.IteratedLocalSearchPolicy._compute_costs
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.iterated_local_search.IteratedLocalSearchPolicy.forward

```{autodoc2-docstring} src.models.policies.iterated_local_search.IteratedLocalSearchPolicy.forward
```

````

`````
