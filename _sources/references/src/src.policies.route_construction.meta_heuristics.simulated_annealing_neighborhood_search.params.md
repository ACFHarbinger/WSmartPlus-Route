# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SANSParams <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams
    :summary:
    ```
````

### API

`````{py:class} SANSParams
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams
```

````{py:attribute} engine
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.engine
:type: str
:value: >
   'new'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.engine
```

````

````{py:attribute} T_init
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.T_init
:type: float
:value: >
   75.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.T_init
```

````

````{py:attribute} iterations_per_T
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.iterations_per_T
:type: int
:value: >
   5000

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.iterations_per_T
```

````

````{py:attribute} alpha
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.alpha
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.alpha
```

````

````{py:attribute} T_min
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.T_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.T_min
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.time_limit
```

````

````{py:attribute} perc_bins_can_overflow
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.perc_bins_can_overflow
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.perc_bins_can_overflow
```

````

````{py:attribute} V
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.V
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.V
```

````

````{py:attribute} shift_duration
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.shift_duration
:type: float
:value: >
   28800.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.shift_duration
```

````

````{py:attribute} combination
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.combination
:type: str
:value: >
   'best'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.combination
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.params.SANSParams.to_dict
```

````

`````
