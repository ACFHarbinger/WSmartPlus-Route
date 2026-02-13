# {py:mod}`src.pipeline.features.test.orchestrator`

```{py:module} src.pipeline.features.test.orchestrator
```

```{autodoc2-docstring} src.pipeline.features.test.orchestrator
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`simulator_testing <src.pipeline.features.test.orchestrator.simulator_testing>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.simulator_testing
    :summary:
    ```
* - {py:obj}`_process_display_updates <src.pipeline.features.test.orchestrator._process_display_updates>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._process_display_updates
    :summary:
    ```
* - {py:obj}`_run_parallel_simulations <src.pipeline.features.test.orchestrator._run_parallel_simulations>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._run_parallel_simulations
    :summary:
    ```
* - {py:obj}`_prepare_parallel_task_args <src.pipeline.features.test.orchestrator._prepare_parallel_task_args>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._prepare_parallel_task_args
    :summary:
    ```
* - {py:obj}`_print_execution_info <src.pipeline.features.test.orchestrator._print_execution_info>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._print_execution_info
    :summary:
    ```
* - {py:obj}`_execute_and_monitor_tasks <src.pipeline.features.test.orchestrator._execute_and_monitor_tasks>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._execute_and_monitor_tasks
    :summary:
    ```
* - {py:obj}`_initialize_simulation_display <src.pipeline.features.test.orchestrator._initialize_simulation_display>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._initialize_simulation_display
    :summary:
    ```
* - {py:obj}`_create_result_containers <src.pipeline.features.test.orchestrator._create_result_containers>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._create_result_containers
    :summary:
    ```
* - {py:obj}`_process_task_result <src.pipeline.features.test.orchestrator._process_task_result>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._process_task_result
    :summary:
    ```
* - {py:obj}`_submit_simulation_tasks <src.pipeline.features.test.orchestrator._submit_simulation_tasks>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._submit_simulation_tasks
    :summary:
    ```
* - {py:obj}`_monitor_tasks_until_complete <src.pipeline.features.test.orchestrator._monitor_tasks_until_complete>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._monitor_tasks_until_complete
    :summary:
    ```
* - {py:obj}`_collect_all_task_results <src.pipeline.features.test.orchestrator._collect_all_task_results>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._collect_all_task_results
    :summary:
    ```
* - {py:obj}`_handle_shutdown <src.pipeline.features.test.orchestrator._handle_shutdown>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._handle_shutdown
    :summary:
    ```
* - {py:obj}`_cleanup_multiprocessing_pool <src.pipeline.features.test.orchestrator._cleanup_multiprocessing_pool>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._cleanup_multiprocessing_pool
    :summary:
    ```
* - {py:obj}`_aggregate_final_results <src.pipeline.features.test.orchestrator._aggregate_final_results>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._aggregate_final_results
    :summary:
    ```
````

### API

````{py:function} simulator_testing(opts, data_size, device)
:canonical: src.pipeline.features.test.orchestrator.simulator_testing

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.simulator_testing
```
````

````{py:function} _process_display_updates(display: logic.src.pipeline.callbacks.simulation_display.SimulationDisplay, shared_metrics: dict, log_tmp: dict, last_reported_days: dict, opts: dict, loop_tic: float, counter: typing.Any)
:canonical: src.pipeline.features.test.orchestrator._process_display_updates

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._process_display_updates
```
````

````{py:function} _run_parallel_simulations(opts, device, indices, sample_idx_ls, weights_path, lock, manager, n_cores, task_count, log_file, original_stderr)
:canonical: src.pipeline.features.test.orchestrator._run_parallel_simulations

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._run_parallel_simulations
```
````

````{py:function} _prepare_parallel_task_args(opts, indices, sample_idx_ls)
:canonical: src.pipeline.features.test.orchestrator._prepare_parallel_task_args

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._prepare_parallel_task_args
```
````

````{py:function} _print_execution_info(opts, task_count, n_cores)
:canonical: src.pipeline.features.test.orchestrator._print_execution_info

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._print_execution_info
```
````

````{py:function} _execute_and_monitor_tasks(pool, opts, device, args, weights_path, n_cores, counter, manager, lock)
:canonical: src.pipeline.features.test.orchestrator._execute_and_monitor_tasks

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._execute_and_monitor_tasks
```
````

````{py:function} _initialize_simulation_display(opts)
:canonical: src.pipeline.features.test.orchestrator._initialize_simulation_display

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._initialize_simulation_display
```
````

````{py:function} _create_result_containers(manager, opts)
:canonical: src.pipeline.features.test.orchestrator._create_result_containers

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._create_result_containers
```
````

````{py:function} _process_task_result(result, log_tmp, failed_log)
:canonical: src.pipeline.features.test.orchestrator._process_task_result

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._process_task_result
```
````

````{py:function} _submit_simulation_tasks(pool, opts, device, args, weights_path, n_cores, callback)
:canonical: src.pipeline.features.test.orchestrator._submit_simulation_tasks

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._submit_simulation_tasks
```
````

````{py:function} _monitor_tasks_until_complete(tasks, display, opts, counter, log_tmp)
:canonical: src.pipeline.features.test.orchestrator._monitor_tasks_until_complete

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._monitor_tasks_until_complete
```
````

````{py:function} _collect_all_task_results(tasks)
:canonical: src.pipeline.features.test.orchestrator._collect_all_task_results

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._collect_all_task_results
```
````

````{py:function} _handle_shutdown(pool, original_stderr)
:canonical: src.pipeline.features.test.orchestrator._handle_shutdown

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._handle_shutdown
```
````

````{py:function} _cleanup_multiprocessing_pool(pool)
:canonical: src.pipeline.features.test.orchestrator._cleanup_multiprocessing_pool

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._cleanup_multiprocessing_pool
```
````

````{py:function} _aggregate_final_results(log_tmp, opts, lock)
:canonical: src.pipeline.features.test.orchestrator._aggregate_final_results

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._aggregate_final_results
```
````
