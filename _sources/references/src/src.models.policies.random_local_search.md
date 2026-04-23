# {py:mod}`src.models.policies.random_local_search`

```{py:module} src.models.policies.random_local_search
```

```{autodoc2-docstring} src.models.policies.random_local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RandomLocalSearchPolicy <src.models.policies.random_local_search.RandomLocalSearchPolicy>`
  - ```{autodoc2-docstring} src.models.policies.random_local_search.RandomLocalSearchPolicy
    :summary:
    ```
````

### API

`````{py:class} RandomLocalSearchPolicy(env_name: str, n_iterations: int = 100, op_probs: typing.Optional[typing.Dict[str, float]] = None, seed: int = 42, device: str = 'cpu', **kwargs: typing.Any)
:canonical: src.models.policies.random_local_search.RandomLocalSearchPolicy

Bases: {py:obj}`logic.src.models.common.improvement.policy.ImprovementPolicy`, {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.models.policies.random_local_search.RandomLocalSearchPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.random_local_search.RandomLocalSearchPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.random_local_search.RandomLocalSearchPolicy.forward

```{autodoc2-docstring} src.models.policies.random_local_search.RandomLocalSearchPolicy.forward
```

````

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.random_local_search.RandomLocalSearchPolicy.__getstate__

```{autodoc2-docstring} src.models.policies.random_local_search.RandomLocalSearchPolicy.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.policies.random_local_search.RandomLocalSearchPolicy.__setstate__

```{autodoc2-docstring} src.models.policies.random_local_search.RandomLocalSearchPolicy.__setstate__
```

````

`````
