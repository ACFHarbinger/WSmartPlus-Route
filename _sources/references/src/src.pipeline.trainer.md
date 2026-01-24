# {py:mod}`src.pipeline.trainer`

```{py:module} src.pipeline.trainer
```

```{autodoc2-docstring} src.pipeline.trainer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WSTrainer <src.pipeline.trainer.WSTrainer>`
  - ```{autodoc2-docstring} src.pipeline.trainer.WSTrainer
    :summary:
    ```
````

### API

`````{py:class} WSTrainer(max_epochs: int = 100, accelerator: str = 'auto', devices: typing.Union[int, str] = 'auto', gradient_clip_val: float = 1.0, log_every_n_steps: int = 50, check_val_every_n_epoch: int = 1, project_name: str = 'wsmart-route', experiment_name: typing.Optional[str] = None, callbacks: typing.Optional[list[pytorch_lightning.callbacks.Callback]] = None, logger: typing.Optional[typing.Union[pytorch_lightning.loggers.Logger, bool]] = None, enable_progress_bar: bool = True, model_weights_path: typing.Optional[str] = None, logs_dir: typing.Optional[str] = None, **kwargs)
:canonical: src.pipeline.trainer.WSTrainer

Bases: {py:obj}`pytorch_lightning.Trainer`

```{autodoc2-docstring} src.pipeline.trainer.WSTrainer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.trainer.WSTrainer.__init__
```

````{py:method} _add_default_callbacks(callbacks: list[pytorch_lightning.callbacks.Callback], enable_progress_bar: bool, model_weights_path: typing.Optional[str] = None) -> list[pytorch_lightning.callbacks.Callback]
:canonical: src.pipeline.trainer.WSTrainer._add_default_callbacks

```{autodoc2-docstring} src.pipeline.trainer.WSTrainer._add_default_callbacks
```

````

````{py:method} _create_default_logger(project_name: str, experiment_name: typing.Optional[str], logs_dir: typing.Optional[str] = None) -> pytorch_lightning.loggers.Logger
:canonical: src.pipeline.trainer.WSTrainer._create_default_logger

```{autodoc2-docstring} src.pipeline.trainer.WSTrainer._create_default_logger
```

````

`````
