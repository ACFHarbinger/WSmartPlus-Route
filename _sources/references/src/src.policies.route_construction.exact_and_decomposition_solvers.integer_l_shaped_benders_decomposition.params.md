# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSBDParams <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams
    :summary:
    ```
````

### API

`````{py:class} ILSBDParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams
```

````{py:attribute} time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.time_limit
```

````

````{py:attribute} n_scenarios
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.n_scenarios
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.n_scenarios
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.profit_aware_operators
```

````

````{py:attribute} max_benders_iterations
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.max_benders_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.max_benders_iterations
```

````

````{py:attribute} benders_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.benders_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.benders_gap
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.overflow_penalty
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.overflow_penalty
```

````

````{py:attribute} undervisit_penalty
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.undervisit_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.undervisit_penalty
```

````

````{py:attribute} collection_threshold
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.collection_threshold
:type: float
:value: >
   70.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.collection_threshold
```

````

````{py:attribute} fill_rate_cv
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.fill_rate_cv
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.fill_rate_cv
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.mip_gap
```

````

````{py:attribute} theta_lower_bound
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.theta_lower_bound
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.theta_lower_bound
```

````

````{py:attribute} verbose
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.verbose
```

````

````{py:attribute} max_cuts_per_round
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.max_cuts_per_round
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.max_cuts_per_round
```

````

````{py:attribute} enable_heuristic_rcc_separation
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.enable_heuristic_rcc_separation
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.enable_heuristic_rcc_separation
```

````

````{py:attribute} enable_comb_cuts
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.enable_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.enable_comb_cuts
```

````

````{py:attribute} engine
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.engine
```

````

````{py:attribute} framework
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.framework
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams.to_dict
```

````

`````
