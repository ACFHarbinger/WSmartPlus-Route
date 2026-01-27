# {py:mod}`src.configs.hpo`

```{py:module} src.configs.hpo
```

```{autodoc2-docstring} src.configs.hpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HPOConfig <src.configs.hpo.HPOConfig>`
  - ```{autodoc2-docstring} src.configs.hpo.HPOConfig
    :summary:
    ```
````

### API

`````{py:class} HPOConfig
:canonical: src.configs.hpo.HPOConfig

```{autodoc2-docstring} src.configs.hpo.HPOConfig
```

````{py:attribute} method
:canonical: src.configs.hpo.HPOConfig.method
:type: str
:value: >
   'dehbo'

```{autodoc2-docstring} src.configs.hpo.HPOConfig.method
```

````

````{py:attribute} metric
:canonical: src.configs.hpo.HPOConfig.metric
:type: str
:value: >
   'reward'

```{autodoc2-docstring} src.configs.hpo.HPOConfig.metric
```

````

````{py:attribute} n_trials
:canonical: src.configs.hpo.HPOConfig.n_trials
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.hpo.HPOConfig.n_trials
```

````

````{py:attribute} n_epochs_per_trial
:canonical: src.configs.hpo.HPOConfig.n_epochs_per_trial
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.hpo.HPOConfig.n_epochs_per_trial
```

````

````{py:attribute} num_workers
:canonical: src.configs.hpo.HPOConfig.num_workers
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.hpo.HPOConfig.num_workers
```

````

````{py:attribute} search_space
:canonical: src.configs.hpo.HPOConfig.search_space
:type: typing.Dict[str, typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.hpo.HPOConfig.search_space
```

````

````{py:attribute} hop_range
:canonical: src.configs.hpo.HPOConfig.hop_range
:type: typing.List[float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.hpo.HPOConfig.hop_range
```

````

````{py:attribute} fevals
:canonical: src.configs.hpo.HPOConfig.fevals
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.hpo.HPOConfig.fevals
```

````

````{py:attribute} timeout
:canonical: src.configs.hpo.HPOConfig.timeout
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.hpo.HPOConfig.timeout
```

````

````{py:attribute} n_startup_trials
:canonical: src.configs.hpo.HPOConfig.n_startup_trials
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.hpo.HPOConfig.n_startup_trials
```

````

````{py:attribute} n_warmup_steps
:canonical: src.configs.hpo.HPOConfig.n_warmup_steps
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.hpo.HPOConfig.n_warmup_steps
```

````

````{py:attribute} min_fidelity
:canonical: src.configs.hpo.HPOConfig.min_fidelity
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.hpo.HPOConfig.min_fidelity
```

````

````{py:attribute} max_fidelity
:canonical: src.configs.hpo.HPOConfig.max_fidelity
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.hpo.HPOConfig.max_fidelity
```

````

````{py:attribute} interval_steps
:canonical: src.configs.hpo.HPOConfig.interval_steps
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.hpo.HPOConfig.interval_steps
```

````

````{py:attribute} eta
:canonical: src.configs.hpo.HPOConfig.eta
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.hpo.HPOConfig.eta
```

````

````{py:attribute} indpb
:canonical: src.configs.hpo.HPOConfig.indpb
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.hpo.HPOConfig.indpb
```

````

````{py:attribute} tournsize
:canonical: src.configs.hpo.HPOConfig.tournsize
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.hpo.HPOConfig.tournsize
```

````

````{py:attribute} cxpb
:canonical: src.configs.hpo.HPOConfig.cxpb
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.hpo.HPOConfig.cxpb
```

````

````{py:attribute} mutpb
:canonical: src.configs.hpo.HPOConfig.mutpb
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.hpo.HPOConfig.mutpb
```

````

````{py:attribute} n_pop
:canonical: src.configs.hpo.HPOConfig.n_pop
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.hpo.HPOConfig.n_pop
```

````

````{py:attribute} n_gen
:canonical: src.configs.hpo.HPOConfig.n_gen
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.hpo.HPOConfig.n_gen
```

````

````{py:attribute} cpu_cores
:canonical: src.configs.hpo.HPOConfig.cpu_cores
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.hpo.HPOConfig.cpu_cores
```

````

````{py:attribute} verbose
:canonical: src.configs.hpo.HPOConfig.verbose
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.hpo.HPOConfig.verbose
```

````

````{py:attribute} train_best
:canonical: src.configs.hpo.HPOConfig.train_best
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.hpo.HPOConfig.train_best
```

````

````{py:attribute} local_mode
:canonical: src.configs.hpo.HPOConfig.local_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.hpo.HPOConfig.local_mode
```

````

````{py:attribute} num_samples
:canonical: src.configs.hpo.HPOConfig.num_samples
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.hpo.HPOConfig.num_samples
```

````

````{py:attribute} max_tres
:canonical: src.configs.hpo.HPOConfig.max_tres
:type: int
:value: >
   14

```{autodoc2-docstring} src.configs.hpo.HPOConfig.max_tres
```

````

````{py:attribute} reduction_factor
:canonical: src.configs.hpo.HPOConfig.reduction_factor
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.hpo.HPOConfig.reduction_factor
```

````

````{py:attribute} max_failures
:canonical: src.configs.hpo.HPOConfig.max_failures
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.hpo.HPOConfig.max_failures
```

````

````{py:attribute} grid
:canonical: src.configs.hpo.HPOConfig.grid
:type: typing.List[float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.hpo.HPOConfig.grid
```

````

````{py:attribute} max_conc
:canonical: src.configs.hpo.HPOConfig.max_conc
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.hpo.HPOConfig.max_conc
```

````

`````
