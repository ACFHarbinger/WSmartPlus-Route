# {py:mod}`src.configs.sim`

```{py:module} src.configs.sim
```

```{autodoc2-docstring} src.configs.sim
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimConfig <src.configs.sim.SimConfig>`
  - ```{autodoc2-docstring} src.configs.sim.SimConfig
    :summary:
    ```
````

### API

`````{py:class} SimConfig
:canonical: src.configs.sim.SimConfig

```{autodoc2-docstring} src.configs.sim.SimConfig
```

````{py:attribute} policies
:canonical: src.configs.sim.SimConfig.policies
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.policies
```

````

````{py:attribute} gate_prob_threshold
:canonical: src.configs.sim.SimConfig.gate_prob_threshold
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.sim.SimConfig.gate_prob_threshold
```

````

````{py:attribute} mask_prob_threshold
:canonical: src.configs.sim.SimConfig.mask_prob_threshold
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.sim.SimConfig.mask_prob_threshold
```

````

````{py:attribute} data_distribution
:canonical: src.configs.sim.SimConfig.data_distribution
:type: str
:value: >
   'gamma1'

```{autodoc2-docstring} src.configs.sim.SimConfig.data_distribution
```

````

````{py:attribute} problem
:canonical: src.configs.sim.SimConfig.problem
:type: str
:value: >
   'vrpp'

```{autodoc2-docstring} src.configs.sim.SimConfig.problem
```

````

````{py:attribute} size
:canonical: src.configs.sim.SimConfig.size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.sim.SimConfig.size
```

````

````{py:attribute} days
:canonical: src.configs.sim.SimConfig.days
:type: int
:value: >
   31

```{autodoc2-docstring} src.configs.sim.SimConfig.days
```

````

````{py:attribute} seed
:canonical: src.configs.sim.SimConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.sim.SimConfig.seed
```

````

````{py:attribute} output_dir
:canonical: src.configs.sim.SimConfig.output_dir
:type: str
:value: >
   'output'

```{autodoc2-docstring} src.configs.sim.SimConfig.output_dir
```

````

````{py:attribute} checkpoint_dir
:canonical: src.configs.sim.SimConfig.checkpoint_dir
:type: str
:value: >
   'temp'

```{autodoc2-docstring} src.configs.sim.SimConfig.checkpoint_dir
```

````

````{py:attribute} checkpoint_days
:canonical: src.configs.sim.SimConfig.checkpoint_days
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.sim.SimConfig.checkpoint_days
```

````

````{py:attribute} log_level
:canonical: src.configs.sim.SimConfig.log_level
:type: str
:value: >
   'INFO'

```{autodoc2-docstring} src.configs.sim.SimConfig.log_level
```

````

````{py:attribute} log_file
:canonical: src.configs.sim.SimConfig.log_file
:type: str
:value: >
   'logs/simulation.log'

```{autodoc2-docstring} src.configs.sim.SimConfig.log_file
```

````

````{py:attribute} cpd
:canonical: src.configs.sim.SimConfig.cpd
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.sim.SimConfig.cpd
```

````

````{py:attribute} n_samples
:canonical: src.configs.sim.SimConfig.n_samples
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.sim.SimConfig.n_samples
```

````

````{py:attribute} resume
:canonical: src.configs.sim.SimConfig.resume
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.sim.SimConfig.resume
```

````

````{py:attribute} cpu_cores
:canonical: src.configs.sim.SimConfig.cpu_cores
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.sim.SimConfig.cpu_cores
```

````

````{py:attribute} n_vehicles
:canonical: src.configs.sim.SimConfig.n_vehicles
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.sim.SimConfig.n_vehicles
```

````

````{py:attribute} area
:canonical: src.configs.sim.SimConfig.area
:type: str
:value: >
   'riomaior'

```{autodoc2-docstring} src.configs.sim.SimConfig.area
```

````

````{py:attribute} waste_type
:canonical: src.configs.sim.SimConfig.waste_type
:type: str
:value: >
   'plastic'

```{autodoc2-docstring} src.configs.sim.SimConfig.waste_type
```

````

````{py:attribute} bin_idx_file
:canonical: src.configs.sim.SimConfig.bin_idx_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.bin_idx_file
```

````

````{py:attribute} decode_type
:canonical: src.configs.sim.SimConfig.decode_type
:type: str
:value: >
   'greedy'

```{autodoc2-docstring} src.configs.sim.SimConfig.decode_type
```

````

````{py:attribute} temperature
:canonical: src.configs.sim.SimConfig.temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.sim.SimConfig.temperature
```

````

````{py:attribute} edge_threshold
:canonical: src.configs.sim.SimConfig.edge_threshold
:type: str
:value: >
   '0'

```{autodoc2-docstring} src.configs.sim.SimConfig.edge_threshold
```

````

````{py:attribute} edge_method
:canonical: src.configs.sim.SimConfig.edge_method
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.edge_method
```

````

````{py:attribute} vertex_method
:canonical: src.configs.sim.SimConfig.vertex_method
:type: str
:value: >
   'mmn'

```{autodoc2-docstring} src.configs.sim.SimConfig.vertex_method
```

````

````{py:attribute} distance_method
:canonical: src.configs.sim.SimConfig.distance_method
:type: str
:value: >
   'ogd'

```{autodoc2-docstring} src.configs.sim.SimConfig.distance_method
```

````

````{py:attribute} dm_filepath
:canonical: src.configs.sim.SimConfig.dm_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.dm_filepath
```

````

````{py:attribute} waste_filepath
:canonical: src.configs.sim.SimConfig.waste_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.waste_filepath
```

````

````{py:attribute} noise_mean
:canonical: src.configs.sim.SimConfig.noise_mean
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.sim.SimConfig.noise_mean
```

````

````{py:attribute} noise_variance
:canonical: src.configs.sim.SimConfig.noise_variance
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.sim.SimConfig.noise_variance
```

````

````{py:attribute} run_tsp
:canonical: src.configs.sim.SimConfig.run_tsp
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.sim.SimConfig.run_tsp
```

````

````{py:attribute} two_opt_max_iter
:canonical: src.configs.sim.SimConfig.two_opt_max_iter
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.sim.SimConfig.two_opt_max_iter
```

````

````{py:attribute} cache_regular
:canonical: src.configs.sim.SimConfig.cache_regular
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.sim.SimConfig.cache_regular
```

````

````{py:attribute} no_cuda
:canonical: src.configs.sim.SimConfig.no_cuda
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.sim.SimConfig.no_cuda
```

````

````{py:attribute} no_progress_bar
:canonical: src.configs.sim.SimConfig.no_progress_bar
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.sim.SimConfig.no_progress_bar
```

````

````{py:attribute} server_run
:canonical: src.configs.sim.SimConfig.server_run
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.sim.SimConfig.server_run
```

````

````{py:attribute} env_file
:canonical: src.configs.sim.SimConfig.env_file
:type: str
:value: >
   'vars.env'

```{autodoc2-docstring} src.configs.sim.SimConfig.env_file
```

````

````{py:attribute} gplic_file
:canonical: src.configs.sim.SimConfig.gplic_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.gplic_file
```

````

````{py:attribute} hexlic_file
:canonical: src.configs.sim.SimConfig.hexlic_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.hexlic_file
```

````

````{py:attribute} symkey_name
:canonical: src.configs.sim.SimConfig.symkey_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.symkey_name
```

````

````{py:attribute} gapik_file
:canonical: src.configs.sim.SimConfig.gapik_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.gapik_file
```

````

````{py:attribute} real_time_log
:canonical: src.configs.sim.SimConfig.real_time_log
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.sim.SimConfig.real_time_log
```

````

````{py:attribute} stats_filepath
:canonical: src.configs.sim.SimConfig.stats_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.stats_filepath
```

````

````{py:attribute} model_path
:canonical: src.configs.sim.SimConfig.model_path
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.model_path
```

````

````{py:attribute} config_path
:canonical: src.configs.sim.SimConfig.config_path
:type: typing.Optional[typing.Dict[str, str]]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.config_path
```

````

````{py:attribute} w_length
:canonical: src.configs.sim.SimConfig.w_length
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.sim.SimConfig.w_length
```

````

````{py:attribute} w_waste
:canonical: src.configs.sim.SimConfig.w_waste
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.sim.SimConfig.w_waste
```

````

````{py:attribute} w_overflows
:canonical: src.configs.sim.SimConfig.w_overflows
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.sim.SimConfig.w_overflows
```

````

````{py:attribute} data_dir
:canonical: src.configs.sim.SimConfig.data_dir
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.sim.SimConfig.data_dir
```

````

`````
