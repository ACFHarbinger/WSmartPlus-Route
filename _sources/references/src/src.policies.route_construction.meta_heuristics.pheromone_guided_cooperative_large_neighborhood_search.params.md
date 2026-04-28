# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACOParams <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams
    :summary:
    ```
* - {py:obj}`LNSParams <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams
    :summary:
    ```
* - {py:obj}`PGCLNSParams <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams
    :summary:
    ```
````

### API

`````{py:class} ACOParams
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams
```

````{py:attribute} n_ants
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.n_ants
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.n_ants
```

````

````{py:attribute} k_sparse
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.k_sparse
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.k_sparse
```

````

````{py:attribute} alpha
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.beta
```

````

````{py:attribute} rho
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.rho
```

````

````{py:attribute} q0
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.q0
```

````

````{py:attribute} tau_0
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.tau_0
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.tau_min
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.time_limit
```

````

````{py:attribute} local_search
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.local_search
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.local_search
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.local_search_iterations
```

````

````{py:attribute} elitist_weight
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.elitist_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.elitist_weight
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams.__post_init__
```

````

`````

`````{py:class} LNSParams
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams
```

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.cooling_rate
```

````

````{py:attribute} reaction_factor
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.reaction_factor
```

````

````{py:attribute} min_removal
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.min_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.min_removal
```

````

````{py:attribute} max_removal_pct
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.max_removal_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams.max_removal_pct
```

````

`````

`````{py:class} PGCLNSParams
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams
```

````{py:attribute} population_size
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.population_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.population_size
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.max_iterations
```

````

````{py:attribute} replacement_rate
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.replacement_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.replacement_rate
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.time_limit
```

````

````{py:attribute} aco
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.aco
:type: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.aco
```

````

````{py:attribute} lns
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.lns
:type: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.lns
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams.to_dict
```

````

`````
