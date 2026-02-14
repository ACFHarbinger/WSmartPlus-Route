# {py:mod}`src.configs.tasks.sim`

```{py:module} src.configs.tasks.sim
```

```{autodoc2-docstring} src.configs.tasks.sim
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimConfig <src.configs.tasks.sim.SimConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.sim.SimConfig
    :summary:
    ```
````

### API

`````{py:class} SimConfig
:canonical: src.configs.tasks.sim.SimConfig

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig
```

````{py:attribute} policies
:canonical: src.configs.tasks.sim.SimConfig.policies
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.policies
```

````

````{py:attribute} data_distribution
:canonical: src.configs.tasks.sim.SimConfig.data_distribution
:type: str
:value: >
   'gamma1'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.data_distribution
```

````

````{py:attribute} problem
:canonical: src.configs.tasks.sim.SimConfig.problem
:type: str
:value: >
   'vrpp'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.problem
```

````

````{py:attribute} days
:canonical: src.configs.tasks.sim.SimConfig.days
:type: int
:value: >
   31

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.days
```

````

````{py:attribute} seed
:canonical: src.configs.tasks.sim.SimConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.seed
```

````

````{py:attribute} output_dir
:canonical: src.configs.tasks.sim.SimConfig.output_dir
:type: str
:value: >
   'output'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.output_dir
```

````

````{py:attribute} checkpoint_dir
:canonical: src.configs.tasks.sim.SimConfig.checkpoint_dir
:type: str
:value: >
   'temp'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.checkpoint_dir
```

````

````{py:attribute} checkpoint_days
:canonical: src.configs.tasks.sim.SimConfig.checkpoint_days
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.checkpoint_days
```

````

````{py:attribute} log_level
:canonical: src.configs.tasks.sim.SimConfig.log_level
:type: str
:value: >
   'INFO'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.log_level
```

````

````{py:attribute} log_file
:canonical: src.configs.tasks.sim.SimConfig.log_file
:type: str
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.log_file
```

````

````{py:attribute} n_samples
:canonical: src.configs.tasks.sim.SimConfig.n_samples
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.n_samples
```

````

````{py:attribute} resume
:canonical: src.configs.tasks.sim.SimConfig.resume
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.resume
```

````

````{py:attribute} cpu_cores
:canonical: src.configs.tasks.sim.SimConfig.cpu_cores
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.cpu_cores
```

````

````{py:attribute} n_vehicles
:canonical: src.configs.tasks.sim.SimConfig.n_vehicles
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.n_vehicles
```

````

````{py:attribute} waste_filepath
:canonical: src.configs.tasks.sim.SimConfig.waste_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.waste_filepath
```

````

````{py:attribute} graph
:canonical: src.configs.tasks.sim.SimConfig.graph
:type: src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.graph
```

````

````{py:attribute} noise_mean
:canonical: src.configs.tasks.sim.SimConfig.noise_mean
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.noise_mean
```

````

````{py:attribute} noise_variance
:canonical: src.configs.tasks.sim.SimConfig.noise_variance
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.noise_variance
```

````

````{py:attribute} cache_regular
:canonical: src.configs.tasks.sim.SimConfig.cache_regular
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.cache_regular
```

````

````{py:attribute} no_cuda
:canonical: src.configs.tasks.sim.SimConfig.no_cuda
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.no_cuda
```

````

````{py:attribute} no_progress_bar
:canonical: src.configs.tasks.sim.SimConfig.no_progress_bar
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.no_progress_bar
```

````

````{py:attribute} server_run
:canonical: src.configs.tasks.sim.SimConfig.server_run
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.server_run
```

````

````{py:attribute} env_file
:canonical: src.configs.tasks.sim.SimConfig.env_file
:type: str
:value: >
   'vars.env'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.env_file
```

````

````{py:attribute} gplic_file
:canonical: src.configs.tasks.sim.SimConfig.gplic_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.gplic_file
```

````

````{py:attribute} hexlic_file
:canonical: src.configs.tasks.sim.SimConfig.hexlic_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.hexlic_file
```

````

````{py:attribute} symkey_name
:canonical: src.configs.tasks.sim.SimConfig.symkey_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.symkey_name
```

````

````{py:attribute} gapik_file
:canonical: src.configs.tasks.sim.SimConfig.gapik_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.gapik_file
```

````

````{py:attribute} real_time_log
:canonical: src.configs.tasks.sim.SimConfig.real_time_log
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.real_time_log
```

````

````{py:attribute} stats_filepath
:canonical: src.configs.tasks.sim.SimConfig.stats_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.stats_filepath
```

````

````{py:attribute} data_dir
:canonical: src.configs.tasks.sim.SimConfig.data_dir
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.data_dir
```

````

````{py:attribute} policy_configs
:canonical: src.configs.tasks.sim.SimConfig.policy_configs
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.sim.SimConfig.policy_configs
```

````

`````
