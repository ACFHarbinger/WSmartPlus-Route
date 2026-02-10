# {py:mod}`src.pipeline.callbacks.model_summary`

```{py:module} src.pipeline.callbacks.model_summary
```

```{autodoc2-docstring} src.pipeline.callbacks.model_summary
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelSummaryCallback <src.pipeline.callbacks.model_summary.ModelSummaryCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.model_summary.ModelSummaryCallback
    :summary:
    ```
````

### API

`````{py:class} ModelSummaryCallback
:canonical: src.pipeline.callbacks.model_summary.ModelSummaryCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.model_summary.ModelSummaryCallback
```

````{py:method} on_train_start(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.model_summary.ModelSummaryCallback.on_train_start

```{autodoc2-docstring} src.pipeline.callbacks.model_summary.ModelSummaryCallback.on_train_start
```

````

````{py:method} _print_summary(model: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.model_summary.ModelSummaryCallback._print_summary

```{autodoc2-docstring} src.pipeline.callbacks.model_summary.ModelSummaryCallback._print_summary
```

````

`````
