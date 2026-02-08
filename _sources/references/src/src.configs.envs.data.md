# {py:mod}`src.configs.envs.data`

```{py:module} src.configs.envs.data
```

```{autodoc2-docstring} src.configs.envs.data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataConfig <src.configs.envs.data.DataConfig>`
  - ```{autodoc2-docstring} src.configs.envs.data.DataConfig
    :summary:
    ```
````

### API

`````{py:class} DataConfig
:canonical: src.configs.envs.data.DataConfig

```{autodoc2-docstring} src.configs.envs.data.DataConfig
```

````{py:attribute} name
:canonical: src.configs.envs.data.DataConfig.name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.envs.data.DataConfig.name
```

````

````{py:attribute} filename
:canonical: src.configs.envs.data.DataConfig.filename
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.envs.data.DataConfig.filename
```

````

````{py:attribute} data_dir
:canonical: src.configs.envs.data.DataConfig.data_dir
:type: str
:value: >
   'datasets'

```{autodoc2-docstring} src.configs.envs.data.DataConfig.data_dir
```

````

````{py:attribute} problem
:canonical: src.configs.envs.data.DataConfig.problem
:type: str
:value: >
   'all'

```{autodoc2-docstring} src.configs.envs.data.DataConfig.problem
```

````

````{py:attribute} mu
:canonical: src.configs.envs.data.DataConfig.mu
:type: typing.Optional[typing.List[float]]
:value: >
   None

```{autodoc2-docstring} src.configs.envs.data.DataConfig.mu
```

````

````{py:attribute} sigma
:canonical: src.configs.envs.data.DataConfig.sigma
:type: typing.Any
:value: >
   0.6

```{autodoc2-docstring} src.configs.envs.data.DataConfig.sigma
```

````

````{py:attribute} data_distributions
:canonical: src.configs.envs.data.DataConfig.data_distributions
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.envs.data.DataConfig.data_distributions
```

````

````{py:attribute} dataset_size
:canonical: src.configs.envs.data.DataConfig.dataset_size
:type: int
:value: >
   128000

```{autodoc2-docstring} src.configs.envs.data.DataConfig.dataset_size
```

````

````{py:attribute} num_locs
:canonical: src.configs.envs.data.DataConfig.num_locs
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.envs.data.DataConfig.num_locs
```

````

````{py:attribute} penalty_factor
:canonical: src.configs.envs.data.DataConfig.penalty_factor
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.envs.data.DataConfig.penalty_factor
```

````

````{py:attribute} overwrite
:canonical: src.configs.envs.data.DataConfig.overwrite
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.envs.data.DataConfig.overwrite
```

````

````{py:attribute} seed
:canonical: src.configs.envs.data.DataConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.envs.data.DataConfig.seed
```

````

````{py:attribute} n_epochs
:canonical: src.configs.envs.data.DataConfig.n_epochs
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.envs.data.DataConfig.n_epochs
```

````

````{py:attribute} epoch_start
:canonical: src.configs.envs.data.DataConfig.epoch_start
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.envs.data.DataConfig.epoch_start
```

````

````{py:attribute} dataset_type
:canonical: src.configs.envs.data.DataConfig.dataset_type
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.envs.data.DataConfig.dataset_type
```

````

````{py:attribute} graph
:canonical: src.configs.envs.data.DataConfig.graph
:type: src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.envs.data.DataConfig.graph
```

````

`````
