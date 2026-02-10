# {py:mod}`src.configs.tasks.data`

```{py:module} src.configs.tasks.data
```

```{autodoc2-docstring} src.configs.tasks.data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataConfig <src.configs.tasks.data.DataConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.data.DataConfig
    :summary:
    ```
````

### API

`````{py:class} DataConfig
:canonical: src.configs.tasks.data.DataConfig

```{autodoc2-docstring} src.configs.tasks.data.DataConfig
```

````{py:attribute} name
:canonical: src.configs.tasks.data.DataConfig.name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.name
```

````

````{py:attribute} filename
:canonical: src.configs.tasks.data.DataConfig.filename
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.filename
```

````

````{py:attribute} data_dir
:canonical: src.configs.tasks.data.DataConfig.data_dir
:type: str
:value: >
   'datasets'

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.data_dir
```

````

````{py:attribute} problem
:canonical: src.configs.tasks.data.DataConfig.problem
:type: str
:value: >
   'all'

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.problem
```

````

````{py:attribute} mu
:canonical: src.configs.tasks.data.DataConfig.mu
:type: typing.Optional[typing.List[float]]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.mu
```

````

````{py:attribute} sigma
:canonical: src.configs.tasks.data.DataConfig.sigma
:type: typing.Any
:value: >
   0.6

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.sigma
```

````

````{py:attribute} data_distributions
:canonical: src.configs.tasks.data.DataConfig.data_distributions
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.data_distributions
```

````

````{py:attribute} dataset_size
:canonical: src.configs.tasks.data.DataConfig.dataset_size
:type: int
:value: >
   128000

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.dataset_size
```

````

````{py:attribute} num_locs
:canonical: src.configs.tasks.data.DataConfig.num_locs
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.num_locs
```

````

````{py:attribute} penalty_factor
:canonical: src.configs.tasks.data.DataConfig.penalty_factor
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.penalty_factor
```

````

````{py:attribute} overwrite
:canonical: src.configs.tasks.data.DataConfig.overwrite
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.overwrite
```

````

````{py:attribute} seed
:canonical: src.configs.tasks.data.DataConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.seed
```

````

````{py:attribute} n_epochs
:canonical: src.configs.tasks.data.DataConfig.n_epochs
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.n_epochs
```

````

````{py:attribute} epoch_start
:canonical: src.configs.tasks.data.DataConfig.epoch_start
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.epoch_start
```

````

````{py:attribute} dataset_type
:canonical: src.configs.tasks.data.DataConfig.dataset_type
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.dataset_type
```

````

````{py:attribute} graph
:canonical: src.configs.tasks.data.DataConfig.graph
:type: logic.src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.data.DataConfig.graph
```

````

`````
