# {py:mod}`src.pipeline.simulations.day_context`

```{py:module} src.pipeline.simulations.day_context
```

```{autodoc2-docstring} src.pipeline.simulations.day_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationDayContext <src.pipeline.simulations.day_context.SimulationDayContext>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`set_daily_waste <src.pipeline.simulations.day_context.set_daily_waste>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day_context.set_daily_waste
    :summary:
    ```
* - {py:obj}`get_daily_results <src.pipeline.simulations.day_context.get_daily_results>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day_context.get_daily_results
    :summary:
    ```
* - {py:obj}`run_day <src.pipeline.simulations.day_context.run_day>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day_context.run_day
    :summary:
    ```
````

### API

`````{py:class} SimulationDayContext
:canonical: src.pipeline.simulations.day_context.SimulationDayContext

Bases: {py:obj}`collections.abc.Mapping`

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext
```

````{py:attribute} graph_size
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.graph_size
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.graph_size
```

````

````{py:attribute} full_policy
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.full_policy
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.full_policy
```

````

````{py:attribute} policy
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.policy
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.policy
```

````

````{py:attribute} policy_name
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.policy_name
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.policy_name
```

````

````{py:attribute} bins
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.bins
:type: typing.Optional[logic.src.pipeline.simulations.bins.Bins]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.bins
```

````

````{py:attribute} new_data
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.new_data
:type: typing.Optional[pandas.DataFrame]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.new_data
```

````

````{py:attribute} coords
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.coords
:type: typing.Optional[pandas.DataFrame]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.coords
```

````

````{py:attribute} distance_matrix
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.distance_matrix
:type: typing.Optional[typing.Union[numpy.ndarray, typing.List[typing.List[float]]]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.distance_matrix
```

````

````{py:attribute} distpath_tup
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.distpath_tup
:type: typing.Tuple[typing.Any, ...]
:value: >
   (None, None, None, None)

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.distpath_tup
```

````

````{py:attribute} distancesC
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.distancesC
:type: typing.Optional[numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.distancesC
```

````

````{py:attribute} paths_between_states
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.paths_between_states
:type: typing.Optional[typing.Dict[typing.Tuple[int, int], typing.List[int]]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.paths_between_states
```

````

````{py:attribute} dm_tensor
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.dm_tensor
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.dm_tensor
```

````

````{py:attribute} run_tsp
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.run_tsp
:type: bool
:value: >
   False

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.run_tsp
```

````

````{py:attribute} sample_id
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.sample_id
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.sample_id
```

````

````{py:attribute} overflows
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.overflows
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.overflows
```

````

````{py:attribute} day
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.day
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.day
```

````

````{py:attribute} model_env
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.model_env
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.model_env
```

````

````{py:attribute} model_ls
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.model_ls
:type: typing.Tuple[typing.Any, ...]
:value: >
   (None,)

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.model_ls
```

````

````{py:attribute} n_vehicles
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.n_vehicles
:type: int
:value: >
   1

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.n_vehicles
```

````

````{py:attribute} area
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.area
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.area
```

````

````{py:attribute} realtime_log_path
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.realtime_log_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.realtime_log_path
```

````

````{py:attribute} waste_type
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.waste_type
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.waste_type
```

````

````{py:attribute} current_collection_day
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.current_collection_day
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.current_collection_day
```

````

````{py:attribute} cached
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.cached
:type: typing.Optional[typing.List[int]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.cached
```

````

````{py:attribute} device
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.device
:type: typing.Optional[torch.device]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.device
```

````

````{py:attribute} lock
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.lock
:type: typing.Optional[multiprocessing.synchronize.Lock]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.lock
```

````

````{py:attribute} hrl_manager
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.hrl_manager
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.hrl_manager
```

````

````{py:attribute} gate_prob_threshold
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.gate_prob_threshold
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.gate_prob_threshold
```

````

````{py:attribute} mask_prob_threshold
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.mask_prob_threshold
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.mask_prob_threshold
```

````

````{py:attribute} two_opt_max_iter
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.two_opt_max_iter
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.two_opt_max_iter
```

````

````{py:attribute} config
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.config
:type: typing.Optional[typing.Dict[str, typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.config
```

````

````{py:attribute} w_length
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.w_length
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.w_length
```

````

````{py:attribute} w_waste
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.w_waste
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.w_waste
```

````

````{py:attribute} w_overflows
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.w_overflows
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.w_overflows
```

````

````{py:attribute} engine
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.engine
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.engine
```

````

````{py:attribute} threshold
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.threshold
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.threshold
```

````

````{py:attribute} daily_log
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.daily_log
:type: typing.Optional[typing.Dict[str, typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.daily_log
```

````

````{py:attribute} output_dict
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.output_dict
:type: typing.Optional[typing.Dict[str, typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.output_dict
```

````

````{py:attribute} tour
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.tour
:type: typing.Optional[typing.List[int]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.tour
```

````

````{py:attribute} cost
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.cost
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.cost
```

````

````{py:attribute} profit
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.profit
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.profit
```

````

````{py:attribute} collected
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.collected
:type: typing.Optional[numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.collected
```

````

````{py:attribute} total_collected
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.total_collected
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.total_collected
```

````

````{py:attribute} ncol
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.ncol
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.ncol
```

````

````{py:attribute} new_overflows
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.new_overflows
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.new_overflows
```

````

````{py:attribute} sum_lost
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.sum_lost
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.sum_lost
```

````

````{py:attribute} fill
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.fill
:type: typing.Optional[numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.fill
```

````

````{py:attribute} total_fill
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.total_fill
:type: typing.Optional[numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.total_fill
```

````

````{py:attribute} extra_output
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.extra_output
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.extra_output
```

````

````{py:attribute} must_go
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.must_go
:type: typing.Optional[typing.List[int]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.must_go
```

````

````{py:property} field_names
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.field_names

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.field_names
```

````

````{py:method} __post_init__()
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.__post_init__

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.__post_init__
```

````

````{py:method} __getitem__(key: str) -> typing.Any
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.__getitem__

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.__getitem__
```

````

````{py:method} __setitem__(key: str, value: typing.Any) -> None
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.__setitem__

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.__setitem__
```

````

````{py:method} get(key: str, default: typing.Any = None) -> typing.Any
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.get

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.get
```

````

````{py:method} __iter__()
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.__iter__

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.__iter__
```

````

````{py:method} __len__()
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.__len__

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.__len__
```

````

````{py:method} __contains__(key: object) -> bool
:canonical: src.pipeline.simulations.day_context.SimulationDayContext.__contains__

```{autodoc2-docstring} src.pipeline.simulations.day_context.SimulationDayContext.__contains__
```

````

`````

````{py:function} set_daily_waste(model_data: typing.Dict[str, typing.Any], waste: numpy.ndarray, device: torch.device, fill: typing.Optional[numpy.ndarray] = None) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.day_context.set_daily_waste

```{autodoc2-docstring} src.pipeline.simulations.day_context.set_daily_waste
```
````

````{py:function} get_daily_results(total_collected: float, ncol: int, cost: float, tour: typing.List[int], day: int, new_overflows: int, sum_lost: float, coordinates: pandas.DataFrame, profit: float) -> typing.Dict[str, typing.Union[int, float, typing.List[typing.Union[int, str]]]]
:canonical: src.pipeline.simulations.day_context.get_daily_results

```{autodoc2-docstring} src.pipeline.simulations.day_context.get_daily_results
```
````

````{py:function} run_day(context: src.pipeline.simulations.day_context.SimulationDayContext) -> src.pipeline.simulations.day_context.SimulationDayContext
:canonical: src.pipeline.simulations.day_context.run_day

```{autodoc2-docstring} src.pipeline.simulations.day_context.run_day
```
````
