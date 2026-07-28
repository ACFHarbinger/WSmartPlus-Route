# {py:mod}`src.pipeline.rl.common.baselines.warmup`

```{py:module} src.pipeline.rl.common.baselines.warmup
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WarmupBaseline <src.pipeline.rl.common.baselines.warmup.WarmupBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup.WarmupBaseline
    :summary:
    ```
````

### API

`````{py:class} WarmupBaseline(baseline: src.pipeline.rl.common.baselines.base.Baseline, warmup_epochs: int = 1, bl_warmup_epochs: typing.Optional[int] = None, beta: float = 0.8, exp_beta: typing.Optional[float] = None, **kwargs)
:canonical: src.pipeline.rl.common.baselines.warmup.WarmupBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.base.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup.WarmupBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup.WarmupBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.warmup.WarmupBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup.WarmupBaseline.eval
```

````

````{py:method} unwrap_batch(batch: typing.Any) -> typing.Tuple[typing.Any, typing.Optional[torch.Tensor]]
:canonical: src.pipeline.rl.common.baselines.warmup.WarmupBaseline.unwrap_batch

```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup.WarmupBaseline.unwrap_batch
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.common.baselines.warmup.WarmupBaseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup.WarmupBaseline.epoch_callback
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.common.baselines.warmup.WarmupBaseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.common.baselines.warmup.WarmupBaseline.get_learnable_parameters
```

````

`````
