# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LBBDParams <src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams
    :summary:
    ```
````

### API

`````{py:class} LBBDParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams
```

````{py:attribute} num_days
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.num_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.num_days
```

````

````{py:attribute} stochastic_master
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.stochastic_master
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.stochastic_master
```

````

````{py:attribute} mean_increment
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.mean_increment
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.mean_increment
```

````

````{py:attribute} num_scenarios
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.num_scenarios
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.num_scenarios
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.max_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.max_iterations
```

````

````{py:attribute} benders_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.benders_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.benders_gap
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.time_limit
```

````

````{py:attribute} subproblem_timeout
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.subproblem_timeout
:type: float
:value: >
   20.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.subproblem_timeout
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.mip_gap
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.mip_gap
```

````

````{py:attribute} waste_weight
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.waste_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.waste_weight
```

````

````{py:attribute} cost_weight
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.cost_weight
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.overflow_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.overflow_penalty
```

````

````{py:attribute} use_nogood_cuts
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.use_nogood_cuts
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.use_nogood_cuts
```

````

````{py:attribute} use_optimality_cuts
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.use_optimality_cuts
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.use_optimality_cuts
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.seed
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.params.LBBDParams.to_dict
```

````

`````
