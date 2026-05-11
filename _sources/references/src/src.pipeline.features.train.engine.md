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
* - {py:obj}`_build_stage_config <src.pipeline.features.train.engine._build_stage_config>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine._build_stage_config
    :summary:
    ```
* - {py:obj}`_run_single_stage <src.pipeline.features.train.engine._run_single_stage>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine._run_single_stage
    :summary:
    ```
* - {py:obj}`_run_curriculum_stages <src.pipeline.features.train.engine._run_curriculum_stages>`
  - ```{autodoc2-docstring} src.pipeline.features.train.engine._run_curriculum_stages
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

````{py:function} _build_experiment_name(cfg: typing.Any) -> str
:canonical: src.pipeline.features.train.engine._build_experiment_name

```{autodoc2-docstring} src.pipeline.features.train.engine._build_experiment_name
```
````

````{py:function} _build_callbacks(cfg: typing.Any) -> list
:canonical: src.pipeline.features.train.engine._build_callbacks

```{autodoc2-docstring} src.pipeline.features.train.engine._build_callbacks
```
````

````{py:function} _build_stage_config(cfg: typing.Any, graph_cfg: typing.Any, stage_idx: typing.Optional[int] = None) -> typing.Any
:canonical: src.pipeline.features.train.engine._build_stage_config

```{autodoc2-docstring} src.pipeline.features.train.engine._build_stage_config
```
````

````{py:function} _run_single_stage(cfg: typing.Any, sinks: typing.Optional[typing.List[typing.Any]], prev_state_dict: typing.Optional[typing.Dict[str, typing.Any]] = None, save_final: bool = True) -> typing.Tuple[float, typing.Optional[typing.Dict[str, typing.Any]]]
:canonical: src.pipeline.features.train.engine._run_single_stage

```{autodoc2-docstring} src.pipeline.features.train.engine._run_single_stage
```
````

````{py:function} _run_curriculum_stages(cfg: typing.Any, sinks: typing.Optional[typing.List[typing.Any]], curriculum_graphs: list) -> float
:canonical: src.pipeline.features.train.engine._run_curriculum_stages

```{autodoc2-docstring} src.pipeline.features.train.engine._run_curriculum_stages
```
````

````{py:function} run_training(cfg: logic.src.configs.Config, sinks: typing.Optional[typing.List[typing.Any]] = None) -> float
:canonical: src.pipeline.features.train.engine.run_training

```{autodoc2-docstring} src.pipeline.features.train.engine.run_training
```
````

````{py:function} _log_training_params(run: logic.src.tracking.Run, cfg: typing.Any) -> None
:canonical: src.pipeline.features.train.engine._log_training_params

```{autodoc2-docstring} src.pipeline.features.train.engine._log_training_params
```
````

````{py:function} _run_training_via_zenml(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.engine._run_training_via_zenml

```{autodoc2-docstring} src.pipeline.features.train.engine._run_training_via_zenml
```
````
