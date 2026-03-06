# {py:mod}`src.models.common.improvement.policy`

```{py:module} src.models.common.improvement.policy
```

```{autodoc2-docstring} src.models.common.improvement.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovementPolicy <src.models.common.improvement.policy.ImprovementPolicy>`
  - ```{autodoc2-docstring} src.models.common.improvement.policy.ImprovementPolicy
    :summary:
    ```
````

### API

`````{py:class} ImprovementPolicy(encoder: typing.Optional[src.models.common.improvement.encoder.ImprovementEncoder] = None, decoder: typing.Optional[src.models.common.improvement.decoder.ImprovementDecoder] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, seed: int = 42, device: str = 'cpu', **kwargs)
:canonical: src.models.common.improvement.policy.ImprovementPolicy

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.improvement.policy.ImprovementPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.improvement.policy.ImprovementPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.common.improvement.policy.ImprovementPolicy.forward

```{autodoc2-docstring} src.models.common.improvement.policy.ImprovementPolicy.forward
```

````

`````
