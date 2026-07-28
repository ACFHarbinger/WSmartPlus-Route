# {py:mod}`src.configs.policies.hmlns`

```{py:module} src.configs.policies.hmlns
```

```{autodoc2-docstring} src.configs.policies.hmlns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HybridMemeticLargeNeighborhoodSearchConfig <src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig
    :summary:
    ```
````

### API

`````{py:class} HybridMemeticLargeNeighborhoodSearchConfig
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig
```

````{py:attribute} population_size
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.population_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.population_size
```

````

````{py:attribute} max_generations
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.max_generations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.max_generations
```

````

````{py:attribute} substitution_rate
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.substitution_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.substitution_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.mutation_rate
```

````

````{py:attribute} elitism_count
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.elitism_count
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.elitism_count
```

````

````{py:attribute} aco_init_iterations
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.aco_init_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.aco_init_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.seed
```

````

````{py:attribute} alns
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.alns
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.alns
```

````

````{py:attribute} maco
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.maco
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.maco
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hmlns.HybridMemeticLargeNeighborhoodSearchConfig.route_improvement
```

````

`````
