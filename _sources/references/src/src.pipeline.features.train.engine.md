# {py:mod}`src.pipeline.features.train.engine`

```{py:module} src.pipeline.features.train.engine
```

```{autodoc2-docstring} src.pipeline.features.train.engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_build_experiment_name <src.pipeline.features.train.engine._build_experiment_name>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine._build_experiment_name
    :summary:
    ```
* - {py:obj}`_build_callbacks <src.pipeline.features.train.engine._build_callbacks>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine._build_callbacks
    :summary:
    ```
* - {py:obj}`run_training <src.pipeline.features.train.engine.run_training>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine.run_training
    :summary:
    ```
* - {py:obj}`_log_training_params <src.pipeline.features.train.engine._log_training_params>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine._log_training_params
    :summary:
    ```
* - {py:obj}`_run_training_via_zenml <src.pipeline.features.train.engine._run_training_via_zenml>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine._run_training_via_zenml
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.features.train.engine.logger>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.features.train.engine.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.train.engine.logger
```

````

````{py:function} _build_experiment_name(cfg: logic.src.configs.Config) -> str
:canonical: src.pipeline.features.train.engine._build_experiment_name

```{autodoc2-docstring} src.pipeline.features.train.engine._build_experiment_name
```
````

````{py:function} _build_callbacks(cfg: logic.src.configs.Config) -> list
:canonical: src.pipeline.features.train.engine._build_callbacks

```{autodoc2-docstring} src.pipeline.features.train.engine._build_callbacks
```
````

````{py:function} run_training(cfg: logic.src.configs.Config, sinks: typing.Optional[typing.List[typing.Any]] = None) -> float
:canonical: src.pipeline.features.train.engine.run_training

```{autodoc2-docstring} src.pipeline.features.train.engine.run_training
```
````

````{py:function} _log_training_params(run: logic.src.tracking.Run, cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.train.engine._log_training_params

```{autodoc2-docstring} src.pipeline.features.train.engine._log_training_params
```
````

````{py:function} _run_training_via_zenml(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.engine._run_training_via_zenml

```{autodoc2-docstring} src.pipeline.features.train.engine._run_training_via_zenml
```
````
