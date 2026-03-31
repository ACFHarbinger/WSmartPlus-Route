# {py:mod}`src.configs.policies.de`

```{py:module} src.configs.policies.de
```

```{autodoc2-docstring} src.configs.policies.de
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEConfig <src.configs.policies.de.DEConfig>`
  - ```{autodoc2-docstring} src.configs.policies.de.DEConfig
    :summary:
    ```
````

### API

`````{py:class} DEConfig
:canonical: src.configs.policies.de.DEConfig

```{autodoc2-docstring} src.configs.policies.de.DEConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.de.DEConfig.pop_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.de.DEConfig.pop_size
```

````

````{py:attribute} mutation_factor
:canonical: src.configs.policies.de.DEConfig.mutation_factor
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.de.DEConfig.mutation_factor
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.de.DEConfig.crossover_rate
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.de.DEConfig.crossover_rate
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.de.DEConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.de.DEConfig.n_removal
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.de.DEConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.de.DEConfig.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.de.DEConfig.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.de.DEConfig.local_search_iterations
```

````

````{py:attribute} evolution_strategy
:canonical: src.configs.policies.de.DEConfig.evolution_strategy
:type: str
:value: >
   'lamarckian'

```{autodoc2-docstring} src.configs.policies.de.DEConfig.evolution_strategy
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.de.DEConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.de.DEConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.de.DEConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.de.DEConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.de.DEConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.de.DEConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.de.DEConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.de.DEConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.de.DEConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.de.DEConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.de.DEConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.de.DEConfig.post_processing
```

````

`````
