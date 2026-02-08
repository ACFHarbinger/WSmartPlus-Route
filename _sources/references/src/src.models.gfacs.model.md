# {py:mod}`src.models.gfacs.model`

```{py:module} src.models.gfacs.model
```

```{autodoc2-docstring} src.models.gfacs.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GFACS <src.models.gfacs.model.GFACS>`
  - ```{autodoc2-docstring} src.models.gfacs.model.GFACS
    :summary:
    ```
````

### API

`````{py:class} GFACS(env: logic.src.envs.base.RL4COEnvBase, policy: typing.Optional[src.models.gfacs.policy.GFACSPolicy] = None, baseline: str = 'no', train_with_local_search: bool = True, policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, baseline_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, alpha_min: float = 0.5, alpha_max: float = 1.0, alpha_flat_epochs: int = 5, beta_min: float = 1.0, beta_max: float = 1.0, beta_flat_epochs: int = 5, **kwargs)
:canonical: src.models.gfacs.model.GFACS

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.gfacs.model.GFACS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.gfacs.model.GFACS.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, phase: str = 'train', **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.gfacs.model.GFACS.forward

```{autodoc2-docstring} src.models.gfacs.model.GFACS.forward
```

````

````{py:property} alpha
:canonical: src.models.gfacs.model.GFACS.alpha
:type: float

```{autodoc2-docstring} src.models.gfacs.model.GFACS.alpha
```

````

````{py:property} beta
:canonical: src.models.gfacs.model.GFACS.beta
:type: float

```{autodoc2-docstring} src.models.gfacs.model.GFACS.beta
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, batch: tensordict.TensorDict, policy_out: typing.Dict[str, typing.Any], reward: typing.Optional[torch.Tensor] = None, log_likelihood: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.gfacs.model.GFACS.calculate_loss

```{autodoc2-docstring} src.models.gfacs.model.GFACS.calculate_loss
```

````

````{py:method} calculate_log_pb_uniform(actions: torch.Tensor, n_ants: int) -> torch.Tensor
:canonical: src.models.gfacs.model.GFACS.calculate_log_pb_uniform

```{autodoc2-docstring} src.models.gfacs.model.GFACS.calculate_log_pb_uniform
```

````

`````
