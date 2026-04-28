# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MACOParams <src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams
    :summary:
    ```
* - {py:obj}`ALNSParams <src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams
    :summary:
    ```
* - {py:obj}`HybridMemeticLargeNeighborhoodSearchParams <src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams
    :summary:
    ```
````

### API

`````{py:class} MACOParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams
```

````{py:attribute} n_ants
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.n_ants
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.n_ants
```

````

````{py:attribute} k_sparse
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.k_sparse
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.k_sparse
```

````

````{py:attribute} alpha
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.beta
```

````

````{py:attribute} rho
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.rho
```

````

````{py:attribute} scale
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.scale
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.scale
```

````

````{py:attribute} tau_0
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.tau_0
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.tau_min
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.tau_max
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.tau_max
```

````

````{py:attribute} p_best
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.p_best
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.p_best
```

````

````{py:attribute} update_schedule
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.update_schedule
:type: str
:value: >
   'auto'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.update_schedule
```

````

````{py:attribute} ib_phase_length
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.ib_phase_length
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.ib_phase_length
```

````

````{py:attribute} elitist_weight
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.elitist_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.elitist_weight
```

````

````{py:attribute} stagnation_limit
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.stagnation_limit
:type: int
:value: >
   25

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.stagnation_limit
```

````

````{py:attribute} restart_pheromone
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.restart_pheromone
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.restart_pheromone
```

````

````{py:attribute} use_restart_best
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.use_restart_best
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.use_restart_best
```

````

````{py:attribute} local_search
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.local_search
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.local_search
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.local_search_iterations
```

````

````{py:attribute} elite_pool_size
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.elite_pool_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.elite_pool_size
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.seed
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.acceptance_criterion
:type: typing.Optional[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.acceptance_criterion
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.__post_init__
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams.from_config
```

````

`````

`````{py:class} ALNSParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams
```

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.start_temp
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.cooling_rate
```

````

````{py:attribute} reaction_factor
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.reaction_factor
```

````

````{py:attribute} min_removal
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.min_removal
:type: int
:value: >
   4

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.min_removal
```

````

````{py:attribute} max_removal_pct
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.max_removal_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.max_removal_pct
```

````

````{py:attribute} segment_size
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.segment_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.segment_size
```

````

````{py:attribute} noise_factor
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.noise_factor
:type: float
:value: >
   0.025

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.noise_factor
```

````

````{py:attribute} worst_removal_randomness
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.worst_removal_randomness
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.worst_removal_randomness
```

````

````{py:attribute} shaw_randomization
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.shaw_randomization
:type: float
:value: >
   6.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.shaw_randomization
```

````

````{py:attribute} max_removal_cap
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.max_removal_cap
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.max_removal_cap
```

````

````{py:attribute} start_temp_control
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.start_temp_control
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.start_temp_control
```

````

````{py:attribute} xi
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.xi
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.xi
```

````

````{py:attribute} regret_pool
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.regret_pool
:type: str
:value: >
   'regret234'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.regret_pool
```

````

````{py:attribute} sigma_1
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.sigma_1
```

````

````{py:attribute} sigma_2
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.sigma_2
```

````

````{py:attribute} sigma_3
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.sigma_3
:type: float
:value: >
   13.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.sigma_3
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.profit_aware_operators
```

````

````{py:attribute} extended_operators
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.extended_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.extended_operators
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.seed
```

````

````{py:attribute} engine
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.engine
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.acceptance_criterion
:type: typing.Optional[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.acceptance_criterion
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams.from_config
```

````

`````

`````{py:class} HybridMemeticLargeNeighborhoodSearchParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams
```

````{py:attribute} population_size
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.population_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.population_size
```

````

````{py:attribute} max_generations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.max_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.max_generations
```

````

````{py:attribute} substitution_rate
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.substitution_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.substitution_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.mutation_rate
```

````

````{py:attribute} elitism_count
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.elitism_count
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.elitism_count
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.n_removal
```

````

````{py:attribute} aco_init_iterations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.aco_init_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.aco_init_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.profit_aware_operators
```

````

````{py:attribute} aco_params
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.aco_params
:type: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.alns_params
:type: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.alns_params
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.from_config
```

````

````{py:property} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.max_iterations
:type: int

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams.max_iterations
```

````

`````
