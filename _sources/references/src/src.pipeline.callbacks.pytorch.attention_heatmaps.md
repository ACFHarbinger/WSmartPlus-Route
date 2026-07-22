# {py:mod}`src.pipeline.callbacks.pytorch.attention_heatmaps`

```{py:module} src.pipeline.callbacks.pytorch.attention_heatmaps
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionHeatmapCallback <src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback
    :summary:
    ```
````

### API

`````{py:class} AttentionHeatmapCallback(tracking_cfg: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback.__init__
```

````{py:method} _enabled() -> bool
:canonical: src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback._enabled

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback._enabled
```

````

````{py:method} _should_run(epoch: int) -> bool
:canonical: src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback._should_run

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback._should_run
```

````

````{py:method} _tb_writer(trainer: pytorch_lightning.Trainer) -> typing.Any
:canonical: src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback._tb_writer

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback._tb_writer
```

````

````{py:method} on_validation_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback.on_validation_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.attention_heatmaps.AttentionHeatmapCallback.on_validation_epoch_end
```

````

`````
