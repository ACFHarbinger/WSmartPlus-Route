# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BBParams <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams
    :summary:
    ```
````

### API

`````{py:class} BBParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams
```

````{py:attribute} time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.mip_gap
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.seed
```

````

````{py:attribute} branching_strategy
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.branching_strategy
:type: str
:value: >
   'strong'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.branching_strategy
```

````

````{py:attribute} strong_branching_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.strong_branching_limit
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.strong_branching_limit
```

````

````{py:attribute} formulation
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.formulation
:type: str
:value: >
   'dfj'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.formulation
```

````

````{py:attribute} lr_lambda_init
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_lambda_init
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_lambda_init
```

````

````{py:attribute} lr_max_subgradient_iters
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_max_subgradient_iters
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_max_subgradient_iters
```

````

````{py:attribute} lr_subgradient_theta
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_subgradient_theta
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_subgradient_theta
```

````

````{py:attribute} lr_subgradient_time_fraction
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_subgradient_time_fraction
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_subgradient_time_fraction
```

````

````{py:attribute} lr_op_time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_op_time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_op_time_limit
```

````

````{py:attribute} lr_branching_strategy
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_branching_strategy
:type: str
:value: >
   'max_waste'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_branching_strategy
```

````

````{py:attribute} lr_max_bb_nodes
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_max_bb_nodes
:type: int
:value: >
   5000

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.lr_max_bb_nodes
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams.from_config
```

````

`````
