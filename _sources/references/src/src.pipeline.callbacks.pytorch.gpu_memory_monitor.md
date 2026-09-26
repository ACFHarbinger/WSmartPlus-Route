# {py:mod}`src.pipeline.callbacks.pytorch.gpu_memory_monitor`

```{py:module} src.pipeline.callbacks.pytorch.gpu_memory_monitor
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.gpu_memory_monitor
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPUMemoryMonitor <src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor
    :summary:
    ```
````

### API

`````{py:class} GPUMemoryMonitor(log_peak_allocated: bool = True, log_peak_reserved: bool = True, log_summary: bool = False, verbose: bool = False)
:canonical: src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor.__init__
```

````{py:method} on_train_epoch_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor.on_train_epoch_start

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor.on_train_epoch_start
```

````

````{py:method} on_train_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.gpu_memory_monitor.GPUMemoryMonitor.on_train_epoch_end
```

````

`````
