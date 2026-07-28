# {py:mod}`src.models.core.polynet.model`

```{py:module} src.models.core.polynet.model
```

```{autodoc2-docstring} src.models.core.polynet.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolyNet <src.models.core.polynet.model.PolyNet>`
  - ```{autodoc2-docstring} src.models.core.polynet.model.PolyNet
    :summary:
    ```
````

### API

`````{py:class} PolyNet(env: logic.src.envs.base.base.RL4COEnvBase, policy: typing.Optional[src.models.core.polynet.policy.PolyNetPolicy] = None, k: int = 128, val_num_solutions: int = 800, encoder_type: str = 'AM', policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, num_augment: int = 8, **kwargs: typing.Any)
:canonical: src.models.core.polynet.model.PolyNet

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.core.polynet.model.PolyNet
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.polynet.model.PolyNet.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, phase: str = 'train', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.polynet.model.PolyNet.forward

```{autodoc2-docstring} src.models.core.polynet.model.PolyNet.forward
```

````

````{py:method} shared_step(batch: tensordict.TensorDict, batch_idx: int, phase: str) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.polynet.model.PolyNet.shared_step

```{autodoc2-docstring} src.models.core.polynet.model.PolyNet.shared_step
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, batch: tensordict.TensorDict, policy_out: typing.Dict[str, typing.Any], reward: typing.Optional[torch.Tensor] = None, log_likelihood: typing.Optional[torch.Tensor] = None) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.polynet.model.PolyNet.calculate_loss

```{autodoc2-docstring} src.models.core.polynet.model.PolyNet.calculate_loss
```

````

````{py:method} rollout(dataset: typing.Any, batch_size: int = 64, device: typing.Union[str, torch.device] = 'cpu') -> torch.Tensor
:canonical: src.models.core.polynet.model.PolyNet.rollout

```{autodoc2-docstring} src.models.core.polynet.model.PolyNet.rollout
```

````

`````
