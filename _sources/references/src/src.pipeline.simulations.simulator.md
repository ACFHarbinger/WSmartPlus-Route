# {py:mod}`src.pipeline.simulations.simulator`

```{py:module} src.pipeline.simulations.simulator
```

```{autodoc2-docstring} src.pipeline.simulations.simulator
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimpleCounter <src.pipeline.simulations.simulator.SimpleCounter>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator.SimpleCounter
    :summary:
    ```
* - {py:obj}`ProgressUpdater <src.pipeline.simulations.simulator.ProgressUpdater>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator.ProgressUpdater
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`init_single_sim_worker <src.pipeline.simulations.simulator.init_single_sim_worker>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator.init_single_sim_worker
    :summary:
    ```
* - {py:obj}`_initialize_worker_repository <src.pipeline.simulations.simulator._initialize_worker_repository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator._initialize_worker_repository
    :summary:
    ```
* - {py:obj}`display_log_metrics <src.pipeline.simulations.simulator.display_log_metrics>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator.display_log_metrics
    :summary:
    ```
* - {py:obj}`single_simulation <src.pipeline.simulations.simulator.single_simulation>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator.single_simulation
    :summary:
    ```
* - {py:obj}`sequential_simulations <src.pipeline.simulations.simulator.sequential_simulations>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator.sequential_simulations
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_lock <src.pipeline.simulations.simulator._lock>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator._lock
    :summary:
    ```
* - {py:obj}`_counter <src.pipeline.simulations.simulator._counter>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator._counter
    :summary:
    ```
* - {py:obj}`_shared_metrics <src.pipeline.simulations.simulator._shared_metrics>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator._shared_metrics
    :summary:
    ```
* - {py:obj}`_task_count <src.pipeline.simulations.simulator._task_count>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator._task_count
    :summary:
    ```
````

### API

````{py:data} _lock
:canonical: src.pipeline.simulations.simulator._lock
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.simulator._lock
```

````

````{py:data} _counter
:canonical: src.pipeline.simulations.simulator._counter
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.simulator._counter
```

````

````{py:data} _shared_metrics
:canonical: src.pipeline.simulations.simulator._shared_metrics
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.simulator._shared_metrics
```

````

````{py:data} _task_count
:canonical: src.pipeline.simulations.simulator._task_count
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.simulator._task_count
```

````

````{py:function} init_single_sim_worker(lock_from_main: typing.Any, counter_from_main: typing.Any, shared_metrics_from_main: typing.Any = None, log_file: typing.Optional[str] = None, cfg: typing.Optional[logic.src.configs.Config] = None, tracking_uri: typing.Optional[str] = None, tracking_run_id: typing.Optional[str] = None, task_count: int = 0) -> None
:canonical: src.pipeline.simulations.simulator.init_single_sim_worker

```{autodoc2-docstring} src.pipeline.simulations.simulator.init_single_sim_worker
```
````

````{py:function} _initialize_worker_repository(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.simulations.simulator._initialize_worker_repository

```{autodoc2-docstring} src.pipeline.simulations.simulator._initialize_worker_repository
```
````

````{py:function} display_log_metrics(output_dir: str, size: int, n_samples: int, days: int, area: str, policies: typing.List[str], log: typing.Dict[str, typing.Union[typing.List[float], typing.Dict[str, float]]], log_std: typing.Optional[typing.Dict[str, typing.Union[typing.List[float], typing.Dict[str, float]]]] = None, lock: typing.Optional[typing.Any] = None) -> None
:canonical: src.pipeline.simulations.simulator.display_log_metrics

```{autodoc2-docstring} src.pipeline.simulations.simulator.display_log_metrics
```
````

````{py:function} single_simulation(cfg: typing.Union[logic.src.configs.Config, omegaconf.DictConfig], device: torch.device, indices: typing.Any, sample_id: int, pol_id: int, model_weights_path: typing.Optional[str], n_cores: int) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.simulator.single_simulation

```{autodoc2-docstring} src.pipeline.simulations.simulator.single_simulation
```
````

`````{py:class} SimpleCounter(val=0)
:canonical: src.pipeline.simulations.simulator.SimpleCounter

```{autodoc2-docstring} src.pipeline.simulations.simulator.SimpleCounter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.simulator.SimpleCounter.__init__
```

````{py:method} update(n=1)
:canonical: src.pipeline.simulations.simulator.SimpleCounter.update

```{autodoc2-docstring} src.pipeline.simulations.simulator.SimpleCounter.update
```

````

`````

`````{py:class} ProgressUpdater(display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter)
:canonical: src.pipeline.simulations.simulator.ProgressUpdater

```{autodoc2-docstring} src.pipeline.simulations.simulator.ProgressUpdater
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.simulator.ProgressUpdater.__init__
```

````{py:method} set_values(display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter) -> None
:canonical: src.pipeline.simulations.simulator.ProgressUpdater.set_values

```{autodoc2-docstring} src.pipeline.simulations.simulator.ProgressUpdater.set_values
```

````

````{py:method} update(n=1) -> None
:canonical: src.pipeline.simulations.simulator.ProgressUpdater.update

```{autodoc2-docstring} src.pipeline.simulations.simulator.ProgressUpdater.update
```

````

`````

````{py:function} sequential_simulations(cfg: logic.src.configs.Config, device: torch.device, indices_ls: typing.List[typing.Any], sample_idx_ls: typing.List[typing.List[int]], model_weights_path: str, lock: typing.Optional[typing.Any], shared_metrics: typing.Optional[typing.Any] = None, task_count: int = 0) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Optional[typing.Dict[str, typing.Any]], typing.List[typing.Dict[str, typing.Any]]]
:canonical: src.pipeline.simulations.simulator.sequential_simulations

```{autodoc2-docstring} src.pipeline.simulations.simulator.sequential_simulations
```
````
