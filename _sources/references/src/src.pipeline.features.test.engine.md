# {py:mod}`src.pipeline.features.test.engine`

```{py:module} src.pipeline.features.test.engine
```

```{autodoc2-docstring} src.pipeline.features.test.engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_wsr_simulator_test <src.pipeline.features.test.engine.run_wsr_simulator_test>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine.run_wsr_simulator_test
    :summary:
    ```
* - {py:obj}`_validate_sim_config <src.pipeline.features.test.engine._validate_sim_config>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine._validate_sim_config
    :summary:
    ```
* - {py:obj}`_resolve_data_size <src.pipeline.features.test.engine._resolve_data_size>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine._resolve_data_size
    :summary:
    ```
* - {py:obj}`_override_waste_filepath <src.pipeline.features.test.engine._override_waste_filepath>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine._override_waste_filepath
    :summary:
    ```
* - {py:obj}`_ensure_directories <src.pipeline.features.test.engine._ensure_directories>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine._ensure_directories
    :summary:
    ```
* - {py:obj}`_log_sim_params <src.pipeline.features.test.engine._log_sim_params>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine._log_sim_params
    :summary:
    ```
* - {py:obj}`_run_sim_via_zenml <src.pipeline.features.test.engine._run_sim_via_zenml>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine._run_sim_via_zenml
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.features.test.engine.logger>`
  - ```{autodoc2-docstring} src.pipeline.features.test.engine.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.features.test.engine.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.test.engine.logger
```

````

````{py:function} run_wsr_simulator_test(cfg: logic.src.configs.Config, sinks: typing.Optional[typing.List[typing.Any]] = None) -> None
:canonical: src.pipeline.features.test.engine.run_wsr_simulator_test

```{autodoc2-docstring} src.pipeline.features.test.engine.run_wsr_simulator_test
```
````

````{py:function} _validate_sim_config(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.test.engine._validate_sim_config

```{autodoc2-docstring} src.pipeline.features.test.engine._validate_sim_config
```
````

````{py:function} _resolve_data_size(cfg: logic.src.configs.Config) -> int
:canonical: src.pipeline.features.test.engine._resolve_data_size

```{autodoc2-docstring} src.pipeline.features.test.engine._resolve_data_size
```
````

````{py:function} _override_waste_filepath(cfg: logic.src.configs.Config, load_dataset: str) -> None
:canonical: src.pipeline.features.test.engine._override_waste_filepath

```{autodoc2-docstring} src.pipeline.features.test.engine._override_waste_filepath
```
````

````{py:function} _ensure_directories(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.test.engine._ensure_directories

```{autodoc2-docstring} src.pipeline.features.test.engine._ensure_directories
```
````

````{py:function} _log_sim_params(run: logic.src.tracking.Run, cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.test.engine._log_sim_params

```{autodoc2-docstring} src.pipeline.features.test.engine._log_sim_params
```
````

````{py:function} _run_sim_via_zenml(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.test.engine._run_sim_via_zenml

```{autodoc2-docstring} src.pipeline.features.test.engine._run_sim_via_zenml
```
````
