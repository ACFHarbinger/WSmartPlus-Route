# {py:mod}`src.policies.league_championship_algorithm.params`

```{py:module} src.policies.league_championship_algorithm.params
```

```{autodoc2-docstring} src.policies.league_championship_algorithm.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LCAParams <src.policies.league_championship_algorithm.params.LCAParams>`
  - ```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams
    :summary:
    ```
````

### API

`````{py:class} LCAParams
:canonical: src.policies.league_championship_algorithm.params.LCAParams

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams
```

````{py:attribute} n_teams
:canonical: src.policies.league_championship_algorithm.params.LCAParams.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.policies.league_championship_algorithm.params.LCAParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.max_iterations
```

````

````{py:attribute} tolerance_pct
:canonical: src.policies.league_championship_algorithm.params.LCAParams.tolerance_pct
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.tolerance_pct
```

````

````{py:attribute} crossover_prob
:canonical: src.policies.league_championship_algorithm.params.LCAParams.crossover_prob
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.crossover_prob
```

````

````{py:attribute} n_removal
:canonical: src.policies.league_championship_algorithm.params.LCAParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.league_championship_algorithm.params.LCAParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.league_championship_algorithm.params.LCAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.league_championship_algorithm.params.LCAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.league_championship_algorithm.params.LCAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.league_championship_algorithm.params.LCAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.league_championship_algorithm.params.LCAParams
:canonical: src.policies.league_championship_algorithm.params.LCAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.league_championship_algorithm.params.LCAParams.from_config
```

````

`````
