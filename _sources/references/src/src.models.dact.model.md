# {py:mod}`src.models.dact.model`

```{py:module} src.models.dact.model
```

```{autodoc2-docstring} src.models.dact.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DACT <src.models.dact.model.DACT>`
  - ```{autodoc2-docstring} src.models.dact.model.DACT
    :summary:
    ```
````

### API

`````{py:class} DACT(env: logic.src.envs.base.RL4COEnvBase, policy: typing.Optional[src.models.dact.policy.DACTPolicy] = None, policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, baseline: str = 'rollout', **kwargs)
:canonical: src.models.dact.model.DACT

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.dact.model.DACT
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.dact.model.DACT.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, phase: str = 'test', **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.dact.model.DACT.forward

```{autodoc2-docstring} src.models.dact.model.DACT.forward
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, batch: tensordict.TensorDict, policy_out: typing.Dict[str, typing.Any], reward: typing.Optional[torch.Tensor] = None, log_likelihood: typing.Optional[torch.Tensor] = None) -> typing.Dict[str, typing.Any]
:canonical: src.models.dact.model.DACT.calculate_loss

```{autodoc2-docstring} src.models.dact.model.DACT.calculate_loss
```

````

````{py:method} shared_step(batch: tensordict.TensorDict, batch_idx: int, phase: str) -> typing.Dict[str, typing.Any]
:canonical: src.models.dact.model.DACT.shared_step

```{autodoc2-docstring} src.models.dact.model.DACT.shared_step
```

````

`````
