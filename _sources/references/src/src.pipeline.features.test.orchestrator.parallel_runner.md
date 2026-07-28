# {py:mod}`src.pipeline.features.test.orchestrator.parallel_runner`

```{py:module} src.pipeline.features.test.orchestrator.parallel_runner
```

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_parallel_simulations <src.pipeline.features.test.orchestrator.parallel_runner.run_parallel_simulations>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.run_parallel_simulations
    :summary:
    ```
* - {py:obj}`execute_and_monitor_tasks <src.pipeline.features.test.orchestrator.parallel_runner.execute_and_monitor_tasks>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.execute_and_monitor_tasks
    :summary:
    ```
* - {py:obj}`handle_shutdown <src.pipeline.features.test.orchestrator.parallel_runner.handle_shutdown>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.handle_shutdown
    :summary:
    ```
* - {py:obj}`cleanup_multiprocessing_pool <src.pipeline.features.test.orchestrator.parallel_runner.cleanup_multiprocessing_pool>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.cleanup_multiprocessing_pool
    :summary:
    ```
````

### API

````{py:function} run_parallel_simulations(cfg: logic.src.configs.Config, device: typing.Any, indices: typing.List[typing.Any], sample_idx_ls: typing.List[typing.List[int]], weights_path: str, lock: typing.Any, manager: typing.Any, n_cores: int, task_count: int, log_file: typing.Optional[str], original_stderr: typing.Any, shared_metrics: typing.Any, tracking_uri: typing.Optional[str] = None, tracking_run_id: typing.Optional[str] = None) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Optional[typing.Dict[str, typing.Any]], typing.List[typing.Dict[str, typing.Any]]]
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.run_parallel_simulations

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.run_parallel_simulations
```
````

````{py:function} execute_and_monitor_tasks(pool: multiprocessing.pool.Pool, cfg: logic.src.configs.Config, device: typing.Any, args: typing.List[typing.Tuple[typing.Any, ...]], weights_path: str, n_cores: int, counter: typing.Any, manager: typing.Any, lock: typing.Any, shared_metrics: typing.Any) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Optional[typing.Dict[str, typing.Any]], typing.List[typing.Dict[str, typing.Any]]]
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.execute_and_monitor_tasks

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.execute_and_monitor_tasks
```
````

````{py:function} handle_shutdown(pool: multiprocessing.pool.Pool, original_stderr: typing.Any) -> None
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.handle_shutdown

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.handle_shutdown
```
````

````{py:function} cleanup_multiprocessing_pool(pool: multiprocessing.pool.Pool) -> None
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.cleanup_multiprocessing_pool

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.cleanup_multiprocessing_pool
```
````
