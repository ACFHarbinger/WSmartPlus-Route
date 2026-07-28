# {py:mod}`src.tracking.integrations.gradient_tracker`

```{py:module} src.tracking.integrations.gradient_tracker
```

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GradientTrackerCallback <src.tracking.integrations.gradient_tracker.GradientTrackerCallback>`
  - ```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback
    :summary:
    ```
````

### API

`````{py:class} GradientTrackerCallback(log_freq: int = 50, hist_freq: int = 200, norm_type: float = 2.0, log_histograms: bool = True, include_bias: bool = False, layer_filter: typing.Optional[typing.List[str]] = None, prefix: str = 'debug/', max_grad_norm_alert: typing.Optional[float] = 100.0)
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback.__init__
```

````{py:method} _should_track(name: str) -> bool
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback._should_track

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback._should_track
```

````

````{py:method} _grad_norm(param: torch.nn.Parameter) -> typing.Optional[float]
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback._grad_norm

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback._grad_norm
```

````

````{py:method} _detect_logger(trainer: pytorch_lightning.Trainer) -> typing.Optional[str]
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback._detect_logger

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback._detect_logger
```

````

````{py:method} on_after_backward(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback.on_after_backward

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback.on_after_backward
```

````

````{py:method} on_train_batch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int) -> None
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback.on_train_batch_end

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback.on_train_batch_end
```

````

````{py:method} _log_histograms_wandb(pl_module: pytorch_lightning.LightningModule, step: int) -> None
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback._log_histograms_wandb

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback._log_histograms_wandb
```

````

````{py:method} _log_histograms_tensorboard(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, step: int) -> None
:canonical: src.tracking.integrations.gradient_tracker.GradientTrackerCallback._log_histograms_tensorboard

```{autodoc2-docstring} src.tracking.integrations.gradient_tracker.GradientTrackerCallback._log_histograms_tensorboard
```

````

`````
