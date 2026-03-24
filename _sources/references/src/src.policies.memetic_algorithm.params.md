# {py:mod}`src.policies.memetic_algorithm.params`

```{py:module} src.policies.memetic_algorithm.params
```

```{autodoc2-docstring} src.policies.memetic_algorithm.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAParams <src.policies.memetic_algorithm.params.MAParams>`
  - ```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams
    :summary:
    ```
````

### API

`````{py:class} MAParams
:canonical: src.policies.memetic_algorithm.params.MAParams

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams
```

````{py:attribute} pop_size
:canonical: src.policies.memetic_algorithm.params.MAParams.pop_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.pop_size
```

````

````{py:attribute} max_generations
:canonical: src.policies.memetic_algorithm.params.MAParams.max_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.max_generations
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.memetic_algorithm.params.MAParams.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.memetic_algorithm.params.MAParams.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.mutation_rate
```

````

````{py:attribute} local_search_rate
:canonical: src.policies.memetic_algorithm.params.MAParams.local_search_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.local_search_rate
```

````

````{py:attribute} tournament_size
:canonical: src.policies.memetic_algorithm.params.MAParams.tournament_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.tournament_size
```

````

````{py:attribute} n_removal
:canonical: src.policies.memetic_algorithm.params.MAParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.n_removal
```

````

````{py:attribute} time_limit
:canonical: src.policies.memetic_algorithm.params.MAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.policies.memetic_algorithm.params.MAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.memetic_algorithm.params.MAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.memetic_algorithm.params.MAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.memetic_algorithm.params.MAParams
:canonical: src.policies.memetic_algorithm.params.MAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.memetic_algorithm.params.MAParams.from_config
```

````

`````
