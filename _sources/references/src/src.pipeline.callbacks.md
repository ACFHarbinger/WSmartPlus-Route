# {py:mod}`src.pipeline.callbacks`

```{py:module} src.pipeline.callbacks
```

```{autodoc2-docstring} src.pipeline.callbacks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SpeedMonitor <src.pipeline.callbacks.SpeedMonitor>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor
    :summary:
    ```
````

### API

`````{py:class} SpeedMonitor(intra_step_time: bool = True, inter_step_time: bool = True, epoch_time: bool = True, verbose: bool = False)
:canonical: src.pipeline.callbacks.SpeedMonitor

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.__init__
```

````{py:method} on_train_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.SpeedMonitor.on_train_start

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.on_train_start
```

````

````{py:method} on_train_epoch_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.SpeedMonitor.on_train_epoch_start

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.on_train_epoch_start
```

````

````{py:method} on_validation_epoch_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.SpeedMonitor.on_validation_epoch_start

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.on_validation_epoch_start
```

````

````{py:method} on_test_epoch_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.SpeedMonitor.on_test_epoch_start

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.on_test_epoch_start
```

````

````{py:method} on_train_batch_start(trainer: pytorch_lightning.Trainer, *unused_args, **unused_kwargs) -> None
:canonical: src.pipeline.callbacks.SpeedMonitor.on_train_batch_start

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.on_train_batch_start
```

````

````{py:method} on_train_batch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, *unused_args, **unused_kwargs) -> None
:canonical: src.pipeline.callbacks.SpeedMonitor.on_train_batch_end

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.on_train_batch_end
```

````

````{py:method} on_train_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.SpeedMonitor.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor.on_train_epoch_end
```

````

````{py:method} _should_log(trainer) -> bool
:canonical: src.pipeline.callbacks.SpeedMonitor._should_log
:staticmethod:

```{autodoc2-docstring} src.pipeline.callbacks.SpeedMonitor._should_log
```

````

`````
