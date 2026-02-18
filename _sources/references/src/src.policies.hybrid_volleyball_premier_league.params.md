# {py:mod}`src.policies.hybrid_volleyball_premier_league.params`

```{py:module} src.policies.hybrid_volleyball_premier_league.params
```

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HVPLParams <src.policies.hybrid_volleyball_premier_league.params.HVPLParams>`
  - ```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams
    :summary:
    ```
````

### API

`````{py:class} HVPLParams
:canonical: src.policies.hybrid_volleyball_premier_league.params.HVPLParams

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams
```

````{py:attribute} n_teams
:canonical: src.policies.hybrid_volleyball_premier_league.params.HVPLParams.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.policies.hybrid_volleyball_premier_league.params.HVPLParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams.max_iterations
```

````

````{py:attribute} sub_rate
:canonical: src.policies.hybrid_volleyball_premier_league.params.HVPLParams.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams.sub_rate
```

````

````{py:attribute} time_limit
:canonical: src.policies.hybrid_volleyball_premier_league.params.HVPLParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams.time_limit
```

````

````{py:attribute} aco_params
:canonical: src.policies.hybrid_volleyball_premier_league.params.HVPLParams.aco_params
:type: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.hybrid_volleyball_premier_league.params.HVPLParams.alns_params
:type: src.policies.adaptive_large_neighborhood_search.params.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.params.HVPLParams.alns_params
```

````

`````
