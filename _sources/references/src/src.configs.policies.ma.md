# {py:mod}`src.configs.policies.ma`

```{py:module} src.configs.policies.ma
```

```{autodoc2-docstring} src.configs.policies.ma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAConfig <src.configs.policies.ma.MAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ma.MAConfig
    :summary:
    ```
````

### API

`````{py:class} MAConfig
:canonical: src.configs.policies.ma.MAConfig

```{autodoc2-docstring} src.configs.policies.ma.MAConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.ma.MAConfig.pop_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.pop_size
```

````

````{py:attribute} max_generations
:canonical: src.configs.policies.ma.MAConfig.max_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.max_generations
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.ma.MAConfig.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.ma.MAConfig.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.mutation_rate
```

````

````{py:attribute} local_search_rate
:canonical: src.configs.policies.ma.MAConfig.local_search_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.local_search_rate
```

````

````{py:attribute} tournament_size
:canonical: src.configs.policies.ma.MAConfig.tournament_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.tournament_size
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ma.MAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.n_removal
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ma.MAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ma.MAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ma.MAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ma.MAConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ma.MAConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ma.MAConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ma.MAConfig.post_processing
```

````

`````
