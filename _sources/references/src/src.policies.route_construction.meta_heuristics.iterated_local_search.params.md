# {py:mod}`src.policies.route_construction.meta_heuristics.iterated_local_search.params`

```{py:module} src.policies.route_construction.meta_heuristics.iterated_local_search.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSParams <src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams
    :summary:
    ```
````

### API

`````{py:class} ILSParams
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams
```

````{py:attribute} n_restarts
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.n_restarts
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.n_restarts
```

````

````{py:attribute} inner_iterations
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.inner_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.inner_iterations
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.n_llh
```

````

````{py:attribute} perturbation_strength
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.perturbation_strength
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.perturbation_strength
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.profit_aware_operators
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.acceptance_criterion
:type: logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.acceptance_criterion
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams
:canonical: src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.iterated_local_search.params.ILSParams.from_config
```

````

`````
