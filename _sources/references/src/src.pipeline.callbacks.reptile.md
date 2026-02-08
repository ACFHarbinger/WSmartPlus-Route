# {py:mod}`src.pipeline.callbacks.reptile`

```{py:module} src.pipeline.callbacks.reptile
```

```{autodoc2-docstring} src.pipeline.callbacks.reptile
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ReptileCallback <src.pipeline.callbacks.reptile.ReptileCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback
    :summary:
    ```
````

### API

`````{py:class} ReptileCallback(num_tasks: int, alpha: float, alpha_decay: float, min_size: int, max_size: int, sch_bar: float = 0.9, data_type: str = 'size', print_log: bool = True)
:canonical: src.pipeline.callbacks.reptile.ReptileCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback.__init__
```

````{py:method} on_fit_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.reptile.ReptileCallback.on_fit_start

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback.on_fit_start
```

````

````{py:method} on_train_epoch_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.reptile.ReptileCallback.on_train_epoch_start

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback.on_train_epoch_start
```

````

````{py:method} on_train_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.reptile.ReptileCallback.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback.on_train_epoch_end
```

````

````{py:method} _sample_task() -> None
:canonical: src.pipeline.callbacks.reptile.ReptileCallback._sample_task

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback._sample_task
```

````

````{py:method} _load_task(pl_module: pytorch_lightning.LightningModule, task_idx: int = 0) -> None
:canonical: src.pipeline.callbacks.reptile.ReptileCallback._load_task

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback._load_task
```

````

````{py:method} _alpha_scheduler() -> None
:canonical: src.pipeline.callbacks.reptile.ReptileCallback._alpha_scheduler

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback._alpha_scheduler
```

````

````{py:method} _generate_task_set(data_type: str, min_size: int, max_size: int) -> list
:canonical: src.pipeline.callbacks.reptile.ReptileCallback._generate_task_set
:staticmethod:

```{autodoc2-docstring} src.pipeline.callbacks.reptile.ReptileCallback._generate_task_set
```

````

`````
