# {py:mod}`src.configs.policies.ga`

```{py:module} src.configs.policies.ga
```

```{autodoc2-docstring} src.configs.policies.ga
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GAConfig <src.configs.policies.ga.GAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ga.GAConfig
    :summary:
    ```
````

### API

`````{py:class} GAConfig
:canonical: src.configs.policies.ga.GAConfig

```{autodoc2-docstring} src.configs.policies.ga.GAConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.ga.GAConfig.pop_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.pop_size
```

````

````{py:attribute} max_generations
:canonical: src.configs.policies.ga.GAConfig.max_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.max_generations
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.ga.GAConfig.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.ga.GAConfig.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.mutation_rate
```

````

````{py:attribute} tournament_size
:canonical: src.configs.policies.ga.GAConfig.tournament_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.tournament_size
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ga.GAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.n_removal
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ga.GAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ga.GAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ga.GAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ga.GAConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ga.GAConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ga.GAConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ga.GAConfig.route_improvement
```

````

`````
