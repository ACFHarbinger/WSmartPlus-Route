# {py:mod}`src.models.l2d.l2d_model`

```{py:module} src.models.l2d.l2d_model
```

```{autodoc2-docstring} src.models.l2d.l2d_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`L2DModel <src.models.l2d.l2d_model.L2DModel>`
  - ```{autodoc2-docstring} src.models.l2d.l2d_model.L2DModel
    :summary:
    ```
````

### API

`````{py:class} L2DModel(env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, policy: typing.Optional[src.models.l2d.policy.L2DPolicy] = None, policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, baseline: str = 'rollout', **kwargs)
:canonical: src.models.l2d.l2d_model.L2DModel

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.l2d.l2d_model.L2DModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.l2d.l2d_model.L2DModel.__init__
```

````{py:method} forward(td: tensordict.TensorDict, phase: str = 'train', return_actions: bool = False, **kwargs) -> dict
:canonical: src.models.l2d.l2d_model.L2DModel.forward

```{autodoc2-docstring} src.models.l2d.l2d_model.L2DModel.forward
```

````

`````
