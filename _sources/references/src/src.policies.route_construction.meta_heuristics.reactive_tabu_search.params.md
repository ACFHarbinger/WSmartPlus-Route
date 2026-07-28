# {py:mod}`src.policies.route_construction.meta_heuristics.reactive_tabu_search.params`

```{py:module} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RTSParams <src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams
    :summary:
    ```
````

### API

`````{py:class} RTSParams
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams
```

````{py:attribute} initial_tenure
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.initial_tenure
:type: int
:value: >
   7

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.initial_tenure
```

````

````{py:attribute} min_tenure
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.min_tenure
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.min_tenure
```

````

````{py:attribute} max_tenure
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.max_tenure
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.max_tenure
```

````

````{py:attribute} tenure_increase
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.tenure_increase
:type: float
:value: >
   1.5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.tenure_increase
```

````

````{py:attribute} tenure_decrease
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.tenure_decrease
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.tenure_decrease
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.profit_aware_operators
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.acceptance_criterion
:type: typing.Optional[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.acceptance_criterion
```

````

````{py:method} from_config(config: logic.src.configs.policies.RTSConfig) -> src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.params.RTSParams.from_config
```

````

`````
