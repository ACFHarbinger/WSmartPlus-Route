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

`````{py:class} GSPO(use_sequence_normalization: bool = True, **kwargs)
:canonical: src.pipeline.rl.core.gspo.GSPO

Bases: {py:obj}`logic.src.pipeline.rl.core.ppo.PPO`

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO.__init__
```

````{py:method} training_step(batch: tensordict.TensorDict, batch_idx: int)
:canonical: src.pipeline.rl.core.gspo.GSPO.training_step

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO.training_step
```

````

````{py:method} calculate_ratio_gspo(new_log_p: torch.Tensor, old_log_p: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor
:canonical: src.pipeline.rl.core.gspo.GSPO.calculate_ratio_gspo

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO.calculate_ratio_gspo
```

````

````{py:method} calculate_advantages(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor
:canonical: src.pipeline.rl.core.gspo.GSPO.calculate_advantages

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO.calculate_advantages
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Any = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.gspo.GSPO.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.gspo.GSPO.calculate_loss
```

````

`````
