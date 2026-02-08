# {py:mod}`src.models.glop.model`

```{py:module} src.models.glop.model
```

```{autodoc2-docstring} src.models.glop.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GLOP <src.models.glop.model.GLOP>`
  - ```{autodoc2-docstring} src.models.glop.model.GLOP
    :summary:
    ```
````

### API

`````{py:class} GLOP(env: logic.src.envs.base.RL4COEnvBase, policy: typing.Optional[src.models.glop.policy.GLOPPolicy] = None, policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, baseline: str = 'mean', **kwargs)
:canonical: src.models.glop.model.GLOP

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.glop.model.GLOP
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.glop.model.GLOP.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, phase: str = 'train', **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.glop.model.GLOP.forward

```{autodoc2-docstring} src.models.glop.model.GLOP.forward
```

````

````{py:method} shared_step(batch: tensordict.TensorDict, batch_idx: int, phase: str) -> typing.Dict[str, typing.Any]
:canonical: src.models.glop.model.GLOP.shared_step

```{autodoc2-docstring} src.models.glop.model.GLOP.shared_step
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, batch: tensordict.TensorDict, policy_out: typing.Dict[str, typing.Any], reward: typing.Optional[torch.Tensor] = None, log_likelihood: typing.Optional[torch.Tensor] = None) -> typing.Dict[str, typing.Any]
:canonical: src.models.glop.model.GLOP.calculate_loss

```{autodoc2-docstring} src.models.glop.model.GLOP.calculate_loss
```

````

`````
