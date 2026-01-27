# {py:mod}`src.pipeline.rl.core.ppo_stepwise`

```{py:module} src.pipeline.rl.core.ppo_stepwise
```

```{autodoc2-docstring} src.pipeline.rl.core.ppo_stepwise
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PPOStepwise <src.pipeline.rl.core.ppo_stepwise.PPOStepwise>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.ppo_stepwise.PPOStepwise
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.pipeline.rl.core.ppo_stepwise.__all__>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.ppo_stepwise.__all__
    :summary:
    ```
````

### API

`````{py:class} PPOStepwise(env: logic.src.envs.base.RL4COEnvBase, policy: torch.nn.Module, critic: torch.nn.Module, **kwargs)
:canonical: src.pipeline.rl.core.ppo_stepwise.PPOStepwise

Bases: {py:obj}`logic.src.pipeline.rl.core.ppo.PPO`

```{autodoc2-docstring} src.pipeline.rl.core.ppo_stepwise.PPOStepwise
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.ppo_stepwise.PPOStepwise.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.ppo_stepwise.PPOStepwise.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.ppo_stepwise.PPOStepwise.calculate_loss
```

````

`````

````{py:data} __all__
:canonical: src.pipeline.rl.core.ppo_stepwise.__all__
:value: >
   ['PPOStepwise']

```{autodoc2-docstring} src.pipeline.rl.core.ppo_stepwise.__all__
```

````
