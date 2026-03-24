# {py:mod}`src.configs.policies.ma_ts`

```{py:module} src.configs.policies.ma_ts
```

```{autodoc2-docstring} src.configs.policies.ma_ts
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmToleranceBasedSelectionConfig <src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmToleranceBasedSelectionConfig
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.engine
:type: str
:value: >
   'ma_ts'

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.engine
```

````

````{py:attribute} population_size
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.population_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.population_size
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.max_iterations
```

````

````{py:attribute} tolerance_pct
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.tolerance_pct
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.tolerance_pct
```

````

````{py:attribute} recombination_rate
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.recombination_rate
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.recombination_rate
```

````

````{py:attribute} perturbation_strength
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.perturbation_strength
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.perturbation_strength
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.n_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_ts.MemeticAlgorithmToleranceBasedSelectionConfig.post_processing
```

````

`````
