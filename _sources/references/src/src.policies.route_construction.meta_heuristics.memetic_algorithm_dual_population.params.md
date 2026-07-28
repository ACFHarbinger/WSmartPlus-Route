# {py:mod}`src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params`

```{py:module} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmDualPopulationParams <src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmDualPopulationParams
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams
```

````{py:attribute} population_size
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.population_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.population_size
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.max_iterations
```

````

````{py:attribute} diversity_injection_rate
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.diversity_injection_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.diversity_injection_rate
```

````

````{py:attribute} elite_learning_weights
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.elite_learning_weights
:type: typing.Optional[typing.List[float]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.elite_learning_weights
```

````

````{py:attribute} elite_count
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.elite_count
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.elite_count
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.from_config
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.__post_init__
```

````

````{py:property} substitution_rate
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.substitution_rate
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.substitution_rate
```

````

````{py:property} elite_size
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.elite_size
:type: int

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.elite_size
```

````

````{py:property} coaching_weight_1
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.coaching_weight_1
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.coaching_weight_1
```

````

````{py:property} coaching_weight_2
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.coaching_weight_2
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.coaching_weight_2
```

````

````{py:property} coaching_weight_3
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.coaching_weight_3
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams.coaching_weight_3
```

````

`````
