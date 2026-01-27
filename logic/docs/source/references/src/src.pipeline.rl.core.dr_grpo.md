# {py:mod}`src.pipeline.rl.core.dr_grpo`

```{py:module} src.pipeline.rl.core.dr_grpo
```

```{autodoc2-docstring} src.pipeline.rl.core.dr_grpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DRGRPO <src.pipeline.rl.core.dr_grpo.DRGRPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.dr_grpo.DRGRPO
    :summary:
    ```
````

### API

`````{py:class} DRGRPO(critic: torch.nn.Module, ppo_epochs: int = 10, eps_clip: float = 0.2, value_loss_weight: float = 0.5, entropy_weight: float = 0.0, max_grad_norm: float = 0.5, mini_batch_size: int | float = 0.25, **kwargs)
:canonical: src.pipeline.rl.core.dr_grpo.DRGRPO

Bases: {py:obj}`logic.src.pipeline.rl.core.ppo.PPO`

```{autodoc2-docstring} src.pipeline.rl.core.dr_grpo.DRGRPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.dr_grpo.DRGRPO.__init__
```

````{py:method} calculate_advantages(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor
:canonical: src.pipeline.rl.core.dr_grpo.DRGRPO.calculate_advantages

```{autodoc2-docstring} src.pipeline.rl.core.dr_grpo.DRGRPO.calculate_advantages
```

````

````{py:method} calculate_ratio(new_log_p: torch.Tensor, old_log_p: torch.Tensor) -> torch.Tensor
:canonical: src.pipeline.rl.core.dr_grpo.DRGRPO.calculate_ratio

```{autodoc2-docstring} src.pipeline.rl.core.dr_grpo.DRGRPO.calculate_ratio
```

````

`````
