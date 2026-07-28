# {py:mod}`src.policies.route_construction.meta_heuristics.volleyball_premier_league.params`

```{py:module} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VPLParams <src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams
    :summary:
    ```
````

### API

`````{py:class} VPLParams
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams
```

````{py:attribute} n_teams
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.n_teams
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.max_iterations
```

````

````{py:attribute} substitution_rate
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.substitution_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.substitution_rate
```

````

````{py:attribute} coaching_weight_1
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.coaching_weight_1
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.coaching_weight_1
```

````

````{py:attribute} coaching_weight_2
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.coaching_weight_2
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.coaching_weight_2
```

````

````{py:attribute} coaching_weight_3
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.coaching_weight_3
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.coaching_weight_3
```

````

````{py:attribute} elite_size
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.elite_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.elite_size
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.profit_aware_operators
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.params.VPLParams.from_config
```

````

`````
