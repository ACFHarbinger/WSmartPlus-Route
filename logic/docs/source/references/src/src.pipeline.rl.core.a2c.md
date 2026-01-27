# {py:mod}`src.pipeline.rl.core.a2c`

```{py:module} src.pipeline.rl.core.a2c
```

```{autodoc2-docstring} src.pipeline.rl.core.a2c
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`A2C <src.pipeline.rl.core.a2c.A2C>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.a2c.A2C
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.pipeline.rl.core.a2c.__all__>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.a2c.__all__
    :summary:
    ```
````

### API

`````{py:class} A2C(env: logic.src.envs.base.RL4COEnvBase, policy: torch.nn.Module, critic: typing.Optional[torch.nn.Module] = None, actor_optimizer: str = 'adam', actor_lr: float = 0.0001, critic_optimizer: str = 'adam', critic_lr: float = 0.001, entropy_coef: float = 0.01, value_loss_coef: float = 0.5, normalize_advantage: bool = True, **kwargs)
:canonical: src.pipeline.rl.core.a2c.A2C

Bases: {py:obj}`logic.src.pipeline.rl.common.base.RL4COLitModule`

```{autodoc2-docstring} src.pipeline.rl.core.a2c.A2C
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.a2c.A2C.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.a2c.A2C.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.a2c.A2C.calculate_loss
```

````

````{py:method} configure_optimizers()
:canonical: src.pipeline.rl.core.a2c.A2C.configure_optimizers

```{autodoc2-docstring} src.pipeline.rl.core.a2c.A2C.configure_optimizers
```

````

````{py:method} training_step(batch: typing.Any, batch_idx: int) -> torch.Tensor
:canonical: src.pipeline.rl.core.a2c.A2C.training_step

```{autodoc2-docstring} src.pipeline.rl.core.a2c.A2C.training_step
```

````

`````

````{py:data} __all__
:canonical: src.pipeline.rl.core.a2c.__all__
:value: >
   ['A2C']

```{autodoc2-docstring} src.pipeline.rl.core.a2c.__all__
```

````
