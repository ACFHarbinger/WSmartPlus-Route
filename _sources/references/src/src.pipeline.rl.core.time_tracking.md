# {py:mod}`src.pipeline.rl.core.time_tracking`

```{py:module} src.pipeline.rl.core.time_tracking
```

```{autodoc2-docstring} src.pipeline.rl.core.time_tracking
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TimeOptimizedREINFORCE <src.pipeline.rl.core.time_tracking.TimeOptimizedREINFORCE>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.time_tracking.TimeOptimizedREINFORCE
    :summary:
    ```
````

### API

`````{py:class} TimeOptimizedREINFORCE(time_sensitivity: float = 0.0, **kwargs)
:canonical: src.pipeline.rl.core.time_tracking.TimeOptimizedREINFORCE

Bases: {py:obj}`src.pipeline.rl.core.reinforce.REINFORCE`

```{autodoc2-docstring} src.pipeline.rl.core.time_tracking.TimeOptimizedREINFORCE
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.time_tracking.TimeOptimizedREINFORCE.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.time_tracking.TimeOptimizedREINFORCE.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.time_tracking.TimeOptimizedREINFORCE.calculate_loss
```

````

`````
