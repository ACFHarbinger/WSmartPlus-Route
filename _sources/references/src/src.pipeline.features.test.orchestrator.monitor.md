# {py:mod}`src.pipeline.features.test.orchestrator.monitor`

```{py:module} src.pipeline.features.test.orchestrator.monitor
```

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`initialize_simulation_display <src.pipeline.features.test.orchestrator.monitor.initialize_simulation_display>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.initialize_simulation_display
    :summary:
    ```
* - {py:obj}`process_display_updates <src.pipeline.features.test.orchestrator.monitor.process_display_updates>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.process_display_updates
    :summary:
    ```
* - {py:obj}`monitor_tasks_until_complete <src.pipeline.features.test.orchestrator.monitor.monitor_tasks_until_complete>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.monitor_tasks_until_complete
    :summary:
    ```
* - {py:obj}`collect_all_task_results <src.pipeline.features.test.orchestrator.monitor.collect_all_task_results>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.collect_all_task_results
    :summary:
    ```
````

### API

````{py:function} initialize_simulation_display(policies: typing.List[str], n_samples: int, total_days: int) -> typing.Optional[logic.src.pipeline.callbacks.SimulationDisplayCallback]
:canonical: src.pipeline.features.test.orchestrator.monitor.initialize_simulation_display

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.initialize_simulation_display
```
````

````{py:function} process_display_updates(display: logic.src.pipeline.callbacks.SimulationDisplayCallback, shared_metrics: dict, log_tmp: dict, last_reported_days: dict, policies: typing.List[str], loop_tic: float, counter: typing.Any) -> None
:canonical: src.pipeline.features.test.orchestrator.monitor.process_display_updates

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.process_display_updates
```
````

````{py:function} monitor_tasks_until_complete(tasks: list, display: typing.Optional[logic.src.pipeline.callbacks.SimulationDisplayCallback], policies: typing.List[str], shared_metrics: typing.Any, counter: typing.Any, log_tmp: typing.Any) -> None
:canonical: src.pipeline.features.test.orchestrator.monitor.monitor_tasks_until_complete

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.monitor_tasks_until_complete
```
````

````{py:function} collect_all_task_results(tasks: list) -> None
:canonical: src.pipeline.features.test.orchestrator.monitor.collect_all_task_results

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.monitor.collect_all_task_results
```
````
