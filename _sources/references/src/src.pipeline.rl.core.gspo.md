# {py:mod}`src.pipeline.rl.core.gspo`

```{py:module} src.pipeline.rl.core.gspo
```

```{autodoc2-docstring} src.pipeline.rl.core.gspo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GSPO <src.pipeline.rl.core.gspo.GSPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO
    :summary:
    ```
````

### API

`````{py:class} GSPO(critic: torch.nn.Module, ppo_epochs: int = 10, eps_clip: float = 0.2, value_loss_weight: float = 0.5, entropy_weight: float = 0.0, max_grad_norm: float = 0.5, mini_batch_size: int | float = 0.25, **kwargs)
:canonical: src.pipeline.rl.core.gspo.GSPO

Bases: {py:obj}`logic.src.pipeline.rl.core.ppo.PPO`

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO.__init__
```

````{py:method} calculate_ratio(new_log_p: torch.Tensor, old_log_p: torch.Tensor) -> torch.Tensor
:canonical: src.pipeline.rl.core.gspo.GSPO.calculate_ratio

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO.calculate_ratio
```

````

`````
