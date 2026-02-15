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

````{py:function} run_parallel_simulations(opts, device, indices, sample_idx_ls, weights_path, lock, manager, n_cores, task_count, log_file, original_stderr)
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.run_parallel_simulations

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.run_parallel_simulations
```
````

````{py:function} execute_and_monitor_tasks(pool, opts, device, args, weights_path, n_cores, counter, manager, lock)
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.execute_and_monitor_tasks

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.execute_and_monitor_tasks
```
````

````{py:function} handle_shutdown(pool, original_stderr)
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.handle_shutdown

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.handle_shutdown
```
````

````{py:function} cleanup_multiprocessing_pool(pool)
:canonical: src.pipeline.features.test.orchestrator.parallel_runner.cleanup_multiprocessing_pool

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.parallel_runner.cleanup_multiprocessing_pool
```
````
