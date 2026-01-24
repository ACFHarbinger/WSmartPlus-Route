# {py:mod}`src.pipeline.rl.core.sapo`

```{py:module} src.pipeline.rl.core.sapo
```

```{autodoc2-docstring} src.pipeline.rl.core.sapo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAPO <src.pipeline.rl.core.sapo.SAPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.sapo.SAPO
    :summary:
    ```
````

### API

`````{py:class} SAPO(tau_pos: float = 0.1, tau_neg: float = 1.0, **kwargs)
:canonical: src.pipeline.rl.core.sapo.SAPO

Bases: {py:obj}`logic.src.pipeline.rl.core.ppo.PPO`

```{autodoc2-docstring} src.pipeline.rl.core.sapo.SAPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.sapo.SAPO.__init__
```

````{py:method} calculate_actor_loss(ratio: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor
:canonical: src.pipeline.rl.core.sapo.SAPO.calculate_actor_loss

```{autodoc2-docstring} src.pipeline.rl.core.sapo.SAPO.calculate_actor_loss
```

````

`````
