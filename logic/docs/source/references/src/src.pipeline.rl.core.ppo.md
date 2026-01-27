# {py:mod}`src.pipeline.rl.core.ppo`

```{py:module} src.pipeline.rl.core.ppo
```

```{autodoc2-docstring} src.pipeline.rl.core.ppo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PPO <src.pipeline.rl.core.ppo.PPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO
    :summary:
    ```
````

### API

`````{py:class} PPO(critic: torch.nn.Module, ppo_epochs: int = 10, eps_clip: float = 0.2, value_loss_weight: float = 0.5, entropy_weight: float = 0.0, max_grad_norm: float = 0.5, mini_batch_size: int | float = 0.25, **kwargs)
:canonical: src.pipeline.rl.core.ppo.PPO

Bases: {py:obj}`logic.src.pipeline.rl.common.base.RL4COLitModule`

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Any = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.ppo.PPO.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.calculate_loss
```

````

````{py:method} training_step(batch: tensordict.TensorDict, batch_idx: int)
:canonical: src.pipeline.rl.core.ppo.PPO.training_step

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.training_step
```

````

````{py:method} calculate_advantages(rewards, values)
:canonical: src.pipeline.rl.core.ppo.PPO.calculate_advantages

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.calculate_advantages
```

````

````{py:method} calculate_ratio(new_log_p, old_log_p)
:canonical: src.pipeline.rl.core.ppo.PPO.calculate_ratio

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.calculate_ratio
```

````

````{py:method} calculate_actor_loss(ratio, advantage)
:canonical: src.pipeline.rl.core.ppo.PPO.calculate_actor_loss

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.calculate_actor_loss
```

````

````{py:method} calculate_critic_loss(values, rewards)
:canonical: src.pipeline.rl.core.ppo.PPO.calculate_critic_loss

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.calculate_critic_loss
```

````

````{py:method} configure_optimizers()
:canonical: src.pipeline.rl.core.ppo.PPO.configure_optimizers

```{autodoc2-docstring} src.pipeline.rl.core.ppo.PPO.configure_optimizers
```

````

`````
