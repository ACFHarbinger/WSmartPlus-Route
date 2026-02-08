# {py:mod}`src.models.l2d.l2d_ppo_model`

```{py:module} src.models.l2d.l2d_ppo_model
```

```{autodoc2-docstring} src.models.l2d.l2d_ppo_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`L2DPPOModel <src.models.l2d.l2d_ppo_model.L2DPPOModel>`
  - ```{autodoc2-docstring} src.models.l2d.l2d_ppo_model.L2DPPOModel
    :summary:
    ```
````

### API

````{py:class} L2DPPOModel(env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, policy: typing.Optional[src.models.l2d.policy.L2DPolicy] = None, policy_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, critic: typing.Optional[torch.nn.Module] = None, critic_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, **kwargs)
:canonical: src.models.l2d.l2d_ppo_model.L2DPPOModel

Bases: {py:obj}`logic.src.pipeline.rl.core.stepwise_ppo.StepwisePPO`

```{autodoc2-docstring} src.models.l2d.l2d_ppo_model.L2DPPOModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.l2d.l2d_ppo_model.L2DPPOModel.__init__
```

````
