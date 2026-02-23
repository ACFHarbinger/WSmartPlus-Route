# {py:mod}`src.tracking.helpers.lightning_helpers`

```{py:module} src.tracking.helpers.lightning_helpers
```

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_opt <src.tracking.helpers.lightning_helpers._opt>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers._opt
    :summary:
    ```
* - {py:obj}`register_monitoring_hooks <src.tracking.helpers.lightning_helpers.register_monitoring_hooks>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.register_monitoring_hooks
    :summary:
    ```
* - {py:obj}`log_hook_stats_to_run <src.tracking.helpers.lightning_helpers.log_hook_stats_to_run>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.log_hook_stats_to_run
    :summary:
    ```
* - {py:obj}`remove_monitoring_hooks <src.tracking.helpers.lightning_helpers.remove_monitoring_hooks>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.remove_monitoring_hooks
    :summary:
    ```
* - {py:obj}`run_periodic_visualisations <src.tracking.helpers.lightning_helpers.run_periodic_visualisations>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.run_periodic_visualisations
    :summary:
    ```
* - {py:obj}`log_execution_profiling_report <src.tracking.helpers.lightning_helpers.log_execution_profiling_report>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.log_execution_profiling_report
    :summary:
    ```
* - {py:obj}`extract_metrics <src.tracking.helpers.lightning_helpers.extract_metrics>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.extract_metrics
    :summary:
    ```
* - {py:obj}`log_checkpoint_artifact <src.tracking.helpers.lightning_helpers.log_checkpoint_artifact>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.log_checkpoint_artifact
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.tracking.helpers.lightning_helpers.logger>`
  - ```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.tracking.helpers.lightning_helpers.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.logger
```

````

````{py:function} _opt(cfg: typing.Optional[logic.src.configs.tracking.TrackingConfig], attr: str, default: typing.Any = False) -> typing.Any
:canonical: src.tracking.helpers.lightning_helpers._opt

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers._opt
```
````

````{py:function} register_monitoring_hooks(cfg: typing.Optional[logic.src.configs.tracking.TrackingConfig], pl_module: pytorch_lightning.LightningModule, hook_data: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.helpers.lightning_helpers.register_monitoring_hooks

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.register_monitoring_hooks
```
````

````{py:function} log_hook_stats_to_run(run: typing.Any, epoch: int, hook_data: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.helpers.lightning_helpers.log_hook_stats_to_run

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.log_hook_stats_to_run
```
````

````{py:function} remove_monitoring_hooks(hook_data: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.helpers.lightning_helpers.remove_monitoring_hooks

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.remove_monitoring_hooks
```
````

````{py:function} run_periodic_visualisations(cfg: typing.Optional[logic.src.configs.tracking.TrackingConfig], pl_module: pytorch_lightning.LightningModule, epoch: int) -> None
:canonical: src.tracking.helpers.lightning_helpers.run_periodic_visualisations

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.run_periodic_visualisations
```
````

````{py:function} log_execution_profiling_report() -> None
:canonical: src.tracking.helpers.lightning_helpers.log_execution_profiling_report

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.log_execution_profiling_report
```
````

````{py:function} extract_metrics(callback_metrics: typing.Dict[str, typing.Any], prefix: str) -> typing.Dict[str, float]
:canonical: src.tracking.helpers.lightning_helpers.extract_metrics

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.extract_metrics
```
````

````{py:function} log_checkpoint_artifact(run: typing.Any, cb: pytorch_lightning.callbacks.ModelCheckpoint, epoch: int) -> None
:canonical: src.tracking.helpers.lightning_helpers.log_checkpoint_artifact

```{autodoc2-docstring} src.tracking.helpers.lightning_helpers.log_checkpoint_artifact
```
````
