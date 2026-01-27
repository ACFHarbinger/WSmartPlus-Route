# {py:mod}`src.pipeline.rl.core.ppo_nstep`

```{py:module} src.pipeline.rl.core.ppo_nstep
```

```{autodoc2-docstring} src.pipeline.rl.core.ppo_nstep
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PPOStep <src.pipeline.rl.core.ppo_nstep.PPOStep>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.ppo_nstep.PPOStep
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.pipeline.rl.core.ppo_nstep.__all__>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.ppo_nstep.__all__
    :summary:
    ```
````

### API

`````{py:class} PPOStep(env: logic.src.envs.base.RL4COEnvBase, policy: torch.nn.Module, critic: torch.nn.Module, n_steps: int = 5, gamma: float = 0.99, **kwargs)
:canonical: src.pipeline.rl.core.ppo_nstep.PPOStep

Bases: {py:obj}`logic.src.pipeline.rl.core.ppo.PPO`

```{autodoc2-docstring} src.pipeline.rl.core.ppo_nstep.PPOStep
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.ppo_nstep.PPOStep.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.ppo_nstep.PPOStep.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.ppo_nstep.PPOStep.calculate_loss
```

````

`````

````{py:data} __all__
:canonical: src.pipeline.rl.core.ppo_nstep.__all__
:value: >
   ['PPOStep']

```{autodoc2-docstring} src.pipeline.rl.core.ppo_nstep.__all__
```

````
