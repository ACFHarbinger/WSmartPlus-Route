# {py:mod}`src.pipeline.callbacks.pytorch.training_health`

```{py:module} src.pipeline.callbacks.pytorch.training_health
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrainingHealthCallback <src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback
    :summary:
    ```
````

### API

`````{py:class} TrainingHealthCallback(max_grad_norm_threshold: float = 100.0, min_entropy_threshold: float = 0.01, stagnation_epochs: int = 50, stagnation_epsilon: float = 0.001, reward_key: str = 'train/reward', entropy_key: str = 'train/entropy', grad_norm_key: str = 'train/grad_norm', alert_cooldown_epochs: int = 5)
:canonical: src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback.__init__
```

````{py:method} on_train_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback.on_train_start

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback.on_train_start
```

````

````{py:method} on_train_batch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int) -> None
:canonical: src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback.on_train_batch_end

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback.on_train_batch_end
```

````

````{py:method} on_train_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback.on_train_epoch_end
```

````

````{py:method} _raise_alert(trainer: pytorch_lightning.Trainer, code: str, severity: str, details: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback._raise_alert

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback._raise_alert
```

````

````{py:method} _metric_value(metrics: typing.Dict[str, typing.Any], key: str) -> typing.Optional[float]
:canonical: src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback._metric_value
:staticmethod:

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback._metric_value
```

````

`````
