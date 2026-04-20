# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HVPLParams <src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams
    :summary:
    ```
````

### API

`````{py:class} HVPLParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams
```

````{py:attribute} n_teams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.n_teams
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.max_iterations
```

````

````{py:attribute} substitution_rate
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.substitution_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.substitution_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.mutation_rate
```

````

````{py:attribute} elite_size
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.elite_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.elite_size
```

````

````{py:attribute} aco_init_iterations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.aco_init_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.aco_init_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.profit_aware_operators
```

````

````{py:attribute} aco_params
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.aco_params
:type: typing.Optional[logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params.KSACOParams]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.alns_params
:type: typing.Optional[logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params.ALNSParams]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.alns_params
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_volleyball_premier_league.params.HVPLParams.from_config
```

````

`````
