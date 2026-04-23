# {py:mod}`src.models.core.glop.model`

```{py:module} src.models.core.glop.model
```

```{autodoc2-docstring} src.models.core.glop.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GLOP <src.models.core.glop.model.GLOP>`
  - ```{autodoc2-docstring} src.models.core.glop.model.GLOP
    :summary:
    ```
````

### API

`````{py:class} GLOP(env: logic.src.envs.base.base.RL4COEnvBase, policy: typing.Optional[src.models.core.glop.policy.GLOPPolicy] = None, policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, baseline: str = 'mean', **kwargs: typing.Any)
:canonical: src.models.core.glop.model.GLOP

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.core.glop.model.GLOP
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.glop.model.GLOP.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, phase: str = 'train', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.glop.model.GLOP.forward

```{autodoc2-docstring} src.models.core.glop.model.GLOP.forward
```

````

````{py:method} shared_step(batch: tensordict.TensorDict, batch_idx: int, phase: str) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.glop.model.GLOP.shared_step

```{autodoc2-docstring} src.models.core.glop.model.GLOP.shared_step
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, batch: tensordict.TensorDict, policy_out: typing.Dict[str, typing.Any], reward: typing.Optional[torch.Tensor] = None, log_likelihood: typing.Optional[torch.Tensor] = None) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.glop.model.GLOP.calculate_loss

```{autodoc2-docstring} src.models.core.glop.model.GLOP.calculate_loss
```

````

`````
