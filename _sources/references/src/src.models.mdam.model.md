# {py:mod}`src.models.mdam.model`

```{py:module} src.models.mdam.model
```

```{autodoc2-docstring} src.models.mdam.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MDAM <src.models.mdam.model.MDAM>`
  - ```{autodoc2-docstring} src.models.mdam.model.MDAM
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`mdam_rollout <src.models.mdam.model.mdam_rollout>`
  - ```{autodoc2-docstring} src.models.mdam.model.mdam_rollout
    :summary:
    ```
````

### API

````{py:function} mdam_rollout(baseline_self: typing.Any, model: torch.nn.Module, env: logic.src.envs.base.RL4COEnvBase, batch_size: int = 64, device: str = 'cpu', dataset: typing.Any = None) -> torch.Tensor
:canonical: src.models.mdam.model.mdam_rollout

```{autodoc2-docstring} src.models.mdam.model.mdam_rollout
```
````

`````{py:class} MDAM(env: logic.src.envs.base.RL4COEnvBase, policy: typing.Optional[src.models.mdam.policy.MDAMPolicy] = None, baseline: str = 'rollout', policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, baseline_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, kl_weight: float = 0.01)
:canonical: src.models.mdam.model.MDAM

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.mdam.model.MDAM
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.mdam.model.MDAM.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, phase: str = 'train', **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.mdam.model.MDAM.forward

```{autodoc2-docstring} src.models.mdam.model.MDAM.forward
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, batch: tensordict.TensorDict, policy_out: typing.Dict[str, typing.Any], reward: typing.Optional[torch.Tensor] = None, log_likelihood: typing.Optional[torch.Tensor] = None) -> typing.Dict[str, typing.Any]
:canonical: src.models.mdam.model.MDAM.calculate_loss

```{autodoc2-docstring} src.models.mdam.model.MDAM.calculate_loss
```

````

````{py:method} patch_baseline_rollout(baseline: typing.Any) -> None
:canonical: src.models.mdam.model.MDAM.patch_baseline_rollout
:staticmethod:

```{autodoc2-docstring} src.models.mdam.model.MDAM.patch_baseline_rollout
```

````

`````
