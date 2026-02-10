# {py:mod}`src.pipeline.callbacks.training_display`

```{py:module} src.pipeline.callbacks.training_display
```

```{autodoc2-docstring} src.pipeline.callbacks.training_display
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrainingDisplayCallback <src.pipeline.callbacks.training_display.TrainingDisplayCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback
    :summary:
    ```
````

### API

`````{py:class} TrainingDisplayCallback(metric_keys: str | typing.List[str] = 'train/reward', chart_title: str = 'Training Progress', refresh_rate: int = 4, history_length: typing.Optional[int] = None, theme: str = 'dark')
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback.__init__
```

````{py:method} on_train_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_start

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_start
```

````

````{py:method} on_train_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_end

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_end
```

````

````{py:method} _init_layout() -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback._init_layout

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback._init_layout
```

````

````{py:method} _start_live_display() -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback._start_live_display

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback._start_live_display
```

````

````{py:method} _render_layout() -> rich.layout.Layout
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback._render_layout

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback._render_layout
```

````

````{py:method} _generate_chart() -> rich.panel.Panel
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback._generate_chart

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback._generate_chart
```

````

````{py:method} _generate_metrics_table() -> rich.panel.Panel
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback._generate_metrics_table

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback._generate_metrics_table
```

````

````{py:method} on_train_epoch_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_epoch_start

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_epoch_start
```

````

````{py:method} on_train_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_epoch_end
```

````

````{py:method} on_train_batch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, outputs: typing.Any, batch: typing.Any, batch_idx: int) -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_batch_end

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_train_batch_end
```

````

````{py:method} on_validation_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_validation_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.training_display.TrainingDisplayCallback.on_validation_epoch_end
```

````

`````
