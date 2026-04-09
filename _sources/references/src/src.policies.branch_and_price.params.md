# {py:mod}`src.policies.branch_and_price.params`

```{py:module} src.policies.branch_and_price.params
```

```{autodoc2-docstring} src.policies.branch_and_price.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BPParams <src.policies.branch_and_price.params.BPParams>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams
    :summary:
    ```
````

### API

`````{py:class} BPParams
:canonical: src.policies.branch_and_price.params.BPParams

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams
```

````{py:attribute} max_iterations
:canonical: src.policies.branch_and_price.params.BPParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.max_iterations
```

````

````{py:attribute} max_routes_per_iteration
:canonical: src.policies.branch_and_price.params.BPParams.max_routes_per_iteration
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.max_routes_per_iteration
```

````

````{py:attribute} optimality_gap
:canonical: src.policies.branch_and_price.params.BPParams.optimality_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.optimality_gap
```

````

````{py:attribute} branching_strategy
:canonical: src.policies.branch_and_price.params.BPParams.branching_strategy
:type: str
:value: >
   'edge'

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.branching_strategy
```

````

````{py:attribute} max_branch_nodes
:canonical: src.policies.branch_and_price.params.BPParams.max_branch_nodes
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.max_branch_nodes
```

````

````{py:attribute} use_exact_pricing
:canonical: src.policies.branch_and_price.params.BPParams.use_exact_pricing
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.use_exact_pricing
```

````

````{py:attribute} use_ng_routes
:canonical: src.policies.branch_and_price.params.BPParams.use_ng_routes
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.use_ng_routes
```

````

````{py:attribute} ng_neighborhood_size
:canonical: src.policies.branch_and_price.params.BPParams.ng_neighborhood_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.ng_neighborhood_size
```

````

````{py:attribute} tree_search_strategy
:canonical: src.policies.branch_and_price.params.BPParams.tree_search_strategy
:type: str
:value: >
   'best_first'

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.tree_search_strategy
```

````

````{py:attribute} vehicle_limit
:canonical: src.policies.branch_and_price.params.BPParams.vehicle_limit
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.vehicle_limit
```

````

````{py:attribute} cleanup_frequency
:canonical: src.policies.branch_and_price.params.BPParams.cleanup_frequency
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.cleanup_frequency
```

````

````{py:attribute} cleanup_threshold
:canonical: src.policies.branch_and_price.params.BPParams.cleanup_threshold
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.cleanup_threshold
```

````

````{py:attribute} early_termination_gap
:canonical: src.policies.branch_and_price.params.BPParams.early_termination_gap
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.early_termination_gap
```

````

````{py:attribute} multiple_waste_types
:canonical: src.policies.branch_and_price.params.BPParams.multiple_waste_types
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.multiple_waste_types
```

````

````{py:attribute} allow_heuristic_ryan_foster
:canonical: src.policies.branch_and_price.params.BPParams.allow_heuristic_ryan_foster
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.allow_heuristic_ryan_foster
```

````

````{py:attribute} use_ryan_foster
:canonical: src.policies.branch_and_price.params.BPParams.use_ryan_foster
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.use_ryan_foster
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.branch_and_price.params.BPParams
:canonical: src.policies.branch_and_price.params.BPParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.branch_and_price.params.BPParams.from_config
```

````

`````
