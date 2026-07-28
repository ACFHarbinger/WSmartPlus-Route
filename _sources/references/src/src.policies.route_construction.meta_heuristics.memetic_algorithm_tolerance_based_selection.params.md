# {py:mod}`src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params`

```{py:module} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmToleranceBasedSelectionParams <src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmToleranceBasedSelectionParams
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams
```

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.seed
```

````

````{py:attribute} population_size
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.population_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.population_size
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.max_iterations
```

````

````{py:attribute} tolerance_pct
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.tolerance_pct
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.tolerance_pct
```

````

````{py:attribute} recombination_rate
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.recombination_rate
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.recombination_rate
```

````

````{py:attribute} perturbation_strength
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.perturbation_strength
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.perturbation_strength
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.n_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.local_search_iterations
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.profit_aware_operators
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.time_limit
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.from_config
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_tolerance_based_selection.params.MemeticAlgorithmToleranceBasedSelectionParams.__post_init__
```

````

`````
