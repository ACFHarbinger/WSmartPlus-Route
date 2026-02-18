# {py:mod}`src.policies.augmented_hybrid_volleyball_premier_league.params`

```{py:module} src.policies.augmented_hybrid_volleyball_premier_league.params
```

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AHVPLParams <src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams>`
  - ```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams
    :summary:
    ```
````

### API

`````{py:class} AHVPLParams
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams
```

````{py:attribute} n_teams
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.max_iterations
```

````

````{py:attribute} sub_rate
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.sub_rate
```

````

````{py:attribute} time_limit
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.time_limit
```

````

````{py:attribute} hgs_params
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.hgs_params
:type: src.policies.hybrid_genetic_search.params.HGSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.hgs_params
```

````

````{py:attribute} aco_params
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.aco_params
:type: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.alns_params
:type: src.policies.adaptive_large_neighborhood_search.params.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams.alns_params
```

````

`````
