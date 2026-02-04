# {py:mod}`src.pipeline.rl.core.adaptive_imitation`

```{py:module} src.pipeline.rl.core.adaptive_imitation
```

```{autodoc2-docstring} src.pipeline.rl.core.adaptive_imitation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveImitation <src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation
    :summary:
    ```
````

### API

`````{py:class} AdaptiveImitation(expert_policy: typing.Any, il_weight: float = 1.0, il_decay: float = 0.95, patience: int = 5, threshold: float = 0.05, decay_step: int = 1, epsilon: float = 1e-05, **kwargs)
:canonical: src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation

Bases: {py:obj}`logic.src.pipeline.rl.core.reinforce.REINFORCE`

```{autodoc2-docstring} src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation.__init__
```

````{py:method} _sanitize_td(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation._sanitize_td

```{autodoc2-docstring} src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation._sanitize_td
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Any = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation.calculate_loss
```

````

````{py:method} on_train_epoch_end()
:canonical: src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.rl.core.adaptive_imitation.AdaptiveImitation.on_train_epoch_end
```

````

`````
