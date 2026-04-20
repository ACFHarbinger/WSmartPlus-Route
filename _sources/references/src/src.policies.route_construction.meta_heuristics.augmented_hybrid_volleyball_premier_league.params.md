# {py:mod}`src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params`

```{py:module} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AHVPLParams <src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams
    :summary:
    ```
````

### API

`````{py:class} AHVPLParams
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams
```

````{py:attribute} n_teams
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.max_iterations
```

````

````{py:attribute} sub_rate
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.sub_rate
```

````

````{py:attribute} elite_alns_iterations
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.elite_alns_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.elite_alns_iterations
```

````

````{py:attribute} not_coached_alns_iterations
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.not_coached_alns_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.not_coached_alns_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.vrpp
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.profit_aware_operators
```

````

````{py:attribute} hgs_params
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.hgs_params
:type: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params.HGSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.hgs_params
```

````

````{py:attribute} aco_params
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.aco_params
:type: logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params.KSACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.alns_params
:type: logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.alns_params
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.__post_init__
```

````

`````
