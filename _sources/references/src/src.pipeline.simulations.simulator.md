# {py:mod}`src.pipeline.simulations.simulator`

```{py:module} src.pipeline.simulations.simulator
```

```{autodoc2-docstring} src.pipeline.simulations.simulator
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`init_single_sim_worker <src.pipeline.simulations.simulator.init_single_sim_worker>`
  - ```{autodoc2-docstring} src.pipeline.simulations.simulator.init_single_sim_worker
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

````{py:function} init_single_sim_worker(lock_from_main: typing.Any, counter_from_main: typing.Any, shared_metrics_from_main: typing.Any = None, log_file: typing.Optional[str] = None) -> None
:canonical: src.pipeline.simulations.simulator.init_single_sim_worker

```{autodoc2-docstring} src.pipeline.simulations.simulator.init_single_sim_worker
```
````

````{py:function} display_log_metrics(output_dir: str, size: int, n_samples: int, days: int, area: str, policies: typing.List[str], log: typing.Dict[str, typing.Union[typing.List[float], typing.Dict[str, float]]], log_std: typing.Optional[typing.Dict[str, typing.Union[typing.List[float], typing.Dict[str, float]]]] = None, lock: typing.Optional[typing.Any] = None) -> None
:canonical: src.pipeline.simulations.simulator.display_log_metrics

```{autodoc2-docstring} src.pipeline.simulations.simulator.display_log_metrics
```
````

````{py:function} single_simulation(opts: typing.Dict[str, typing.Any], device: torch.device, indices: typing.Any, sample_id: int, pol_id: int, model_weights_path: str, n_cores: int) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.simulator.single_simulation

```{autodoc2-docstring} src.pipeline.simulations.simulator.single_simulation
```
````

````{py:function} sequential_simulations(opts: typing.Dict[str, typing.Any], device: torch.device, indices_ls: typing.List[typing.Any], sample_idx_ls: typing.List[typing.List[int]], model_weights_path: str, lock: typing.Optional[typing.Any]) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Optional[typing.Dict[str, typing.Any]], typing.List[typing.Dict[str, typing.Any]]]
:canonical: src.pipeline.simulations.simulator.sequential_simulations

```{autodoc2-docstring} src.pipeline.simulations.simulator.sequential_simulations
```
````
