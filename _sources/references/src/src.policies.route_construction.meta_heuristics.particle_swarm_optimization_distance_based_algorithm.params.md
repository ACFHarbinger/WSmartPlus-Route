# {py:mod}`src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params`

```{py:module} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistancePSOParams <src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams
    :summary:
    ```
````

### API

`````{py:class} DistancePSOParams
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams
```

````{py:attribute} population_size
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.population_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.population_size
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.max_iterations
```

````

````{py:attribute} inertia_weight_start
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.inertia_weight_start
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.inertia_weight_start
```

````

````{py:attribute} inertia_weight_end
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.inertia_weight_end
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.inertia_weight_end
```

````

````{py:attribute} cognitive_coef
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.cognitive_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.cognitive_coef
```

````

````{py:attribute} social_coef
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.social_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.social_coef
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.n_removal
```

````

````{py:attribute} velocity_to_mutation_rate
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.velocity_to_mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.velocity_to_mutation_rate
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.time_limit
```

````

````{py:attribute} alpha_profit
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.alpha_profit
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.alpha_profit
```

````

````{py:attribute} beta_will
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.beta_will
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.beta_will
```

````

````{py:attribute} gamma_cost
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.gamma_cost
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.gamma_cost
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.seed
```

````

````{py:method} get_inertia_weight(iteration: int) -> float
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.get_inertia_weight

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.get_inertia_weight
```

````

````{py:property} c1
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.c1
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.c1
```

````

````{py:property} c2
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.c2
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.c2
```

````

````{py:property} w_start
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.w_start
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.w_start
```

````

````{py:property} w_end
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.w_end
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.w_end
```

````

````{py:property} pop_size
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.pop_size
:type: int

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.pop_size
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.params.DistancePSOParams.from_config
```

````

`````
