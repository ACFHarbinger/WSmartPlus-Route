# {py:mod}`src.policies.route_construction.meta_heuristics.firefly_algorithm.params`

```{py:module} src.policies.route_construction.meta_heuristics.firefly_algorithm.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FAParams <src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams
    :summary:
    ```
````

### API

`````{py:class} FAParams
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams
```

````{py:attribute} pop_size
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.pop_size
```

````

````{py:attribute} beta0
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.beta0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.beta0
```

````

````{py:attribute} gamma
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.gamma
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.gamma
```

````

````{py:attribute} alpha_profit
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.alpha_profit
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.alpha_profit
```

````

````{py:attribute} beta_will
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.beta_will
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.beta_will
```

````

````{py:attribute} gamma_cost
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.gamma_cost
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.gamma_cost
```

````

````{py:attribute} alpha_rnd
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.alpha_rnd
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.alpha_rnd
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.n_removal
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams
:canonical: src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.firefly_algorithm.params.FAParams.from_config
```

````

`````
