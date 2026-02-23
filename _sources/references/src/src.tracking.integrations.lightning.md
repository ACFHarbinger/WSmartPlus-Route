# {py:mod}`src.tracking.integrations.lightning`

```{py:module} src.tracking.integrations.lightning
```

```{autodoc2-docstring} src.tracking.integrations.lightning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrackingCallback <src.tracking.integrations.lightning.TrackingCallback>`
  - ```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.tracking.integrations.lightning.logger>`
  - ```{autodoc2-docstring} src.tracking.integrations.lightning.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.tracking.integrations.lightning.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.tracking.integrations.lightning.logger
```

````

`````{py:class} TrackingCallback(tracking_cfg: typing.Optional[logic.src.configs.tracking.TrackingConfig] = None)
:canonical: src.tracking.integrations.lightning.TrackingCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback.__init__
```

````{py:method} on_fit_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_fit_start

````

````{py:method} on_train_batch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_train_batch_end

```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback.on_train_batch_end
```

````

````{py:method} on_train_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_train_epoch_end

````

````{py:method} on_validation_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_validation_epoch_end

````

````{py:method} on_train_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_train_end

````

````{py:method} on_before_optimizer_step(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, optimizer: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_before_optimizer_step

```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback.on_before_optimizer_step
```

````

````{py:method} on_save_checkpoint(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, checkpoint: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_save_checkpoint

````

````{py:method} on_train_dataloader(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_train_dataloader

```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback.on_train_dataloader
```

````

`````
