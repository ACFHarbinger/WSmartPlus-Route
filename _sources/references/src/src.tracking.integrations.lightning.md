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

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_extract_metrics <src.tracking.integrations.lightning._extract_metrics>`
  - ```{autodoc2-docstring} src.tracking.integrations.lightning._extract_metrics
    :summary:
    ```
* - {py:obj}`_log_checkpoint_artifact <src.tracking.integrations.lightning._log_checkpoint_artifact>`
  - ```{autodoc2-docstring} src.tracking.integrations.lightning._log_checkpoint_artifact
    :summary:
    ```
````

### API

`````{py:class} TrackingCallback
:canonical: src.tracking.integrations.lightning.TrackingCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback
```

````{py:method} on_fit_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_fit_start

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

````{py:method} on_save_checkpoint(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, checkpoint: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_save_checkpoint

````

````{py:method} on_train_dataloader(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.tracking.integrations.lightning.TrackingCallback.on_train_dataloader

```{autodoc2-docstring} src.tracking.integrations.lightning.TrackingCallback.on_train_dataloader
```

````

`````

````{py:function} _extract_metrics(callback_metrics: typing.Dict[str, typing.Any], prefix: str) -> typing.Dict[str, float]
:canonical: src.tracking.integrations.lightning._extract_metrics

```{autodoc2-docstring} src.tracking.integrations.lightning._extract_metrics
```
````

````{py:function} _log_checkpoint_artifact(run: typing.Any, cb: pytorch_lightning.callbacks.ModelCheckpoint, epoch: int) -> None
:canonical: src.tracking.integrations.lightning._log_checkpoint_artifact

```{autodoc2-docstring} src.tracking.integrations.lightning._log_checkpoint_artifact
```
````
