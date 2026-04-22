# {py:mod}`src.policies.route_construction.meta_heuristics.tabu_search.params`

```{py:module} src.policies.route_construction.meta_heuristics.tabu_search.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSParams <src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams
    :summary:
    ```
````

### API

`````{py:class} TSParams
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams
```

````{py:attribute} tabu_tenure
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.tabu_tenure
:type: int
:value: >
   7

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.tabu_tenure
```

````

````{py:attribute} dynamic_tenure
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.dynamic_tenure
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.dynamic_tenure
```

````

````{py:attribute} min_tenure
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.min_tenure
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.min_tenure
```

````

````{py:attribute} max_tenure
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.max_tenure
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.max_tenure
```

````

````{py:attribute} aspiration_enabled
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.aspiration_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.aspiration_enabled
```

````

````{py:attribute} intensification_enabled
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.intensification_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.intensification_enabled
```

````

````{py:attribute} diversification_enabled
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.diversification_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.diversification_enabled
```

````

````{py:attribute} intensification_interval
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.intensification_interval
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.intensification_interval
```

````

````{py:attribute} diversification_interval
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.diversification_interval
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.diversification_interval
```

````

````{py:attribute} elite_size
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.elite_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.elite_size
```

````

````{py:attribute} frequency_penalty_weight
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.frequency_penalty_weight
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.frequency_penalty_weight
```

````

````{py:attribute} candidate_list_enabled
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.candidate_list_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.candidate_list_enabled
```

````

````{py:attribute} candidate_list_size
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.candidate_list_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.candidate_list_size
```

````

````{py:attribute} oscillation_enabled
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.oscillation_enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.oscillation_enabled
```

````

````{py:attribute} feasibility_tolerance
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.feasibility_tolerance
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.feasibility_tolerance
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.max_iterations
```

````

````{py:attribute} max_iterations_no_improve
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.max_iterations_no_improve
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.max_iterations_no_improve
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.profit_aware_operators
```

````

````{py:attribute} use_swap
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_swap
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_swap
```

````

````{py:attribute} use_relocate
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_relocate
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_relocate
```

````

````{py:attribute} use_2opt
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_2opt
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_2opt
```

````

````{py:attribute} use_insertion
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_insertion
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.use_insertion
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.acceptance_criterion
:type: logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.acceptance_criterion
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams
:canonical: src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.tabu_search.params.TSParams.from_config
```

````

`````
