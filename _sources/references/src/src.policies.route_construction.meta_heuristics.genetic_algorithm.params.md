# {py:mod}`src.policies.route_construction.meta_heuristics.genetic_algorithm.params`

```{py:module} src.policies.route_construction.meta_heuristics.genetic_algorithm.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GAParams <src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams
    :summary:
    ```
````

### API

`````{py:class} GAParams
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams
```

````{py:attribute} pop_size
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.pop_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.pop_size
```

````

````{py:attribute} max_generations
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.max_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.max_generations
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.mutation_rate
```

````

````{py:attribute} tournament_size
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.tournament_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.tournament_size
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.n_removal
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams.from_config
```

````

`````
