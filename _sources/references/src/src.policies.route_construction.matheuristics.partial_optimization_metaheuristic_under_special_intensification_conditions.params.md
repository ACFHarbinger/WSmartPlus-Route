# {py:mod}`src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params`

```{py:module} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`POPMUSICParams <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams
    :summary:
    ```
````

### API

`````{py:class} POPMUSICParams
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams
```

````{py:attribute} subproblem_size
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.subproblem_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.subproblem_size
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.max_iterations
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.max_iterations
```

````

````{py:attribute} base_solver
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.base_solver
:type: str
:value: >
   'fast_tsp'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.base_solver
```

````

````{py:attribute} base_solver_config
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.base_solver_config
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.base_solver_config
```

````

````{py:attribute} cluster_solver
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.cluster_solver
:type: str
:value: >
   'fast_tsp'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.cluster_solver
```

````

````{py:attribute} cluster_solver_config
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.cluster_solver_config
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.cluster_solver_config
```

````

````{py:attribute} initial_solver
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.initial_solver
:type: str
:value: >
   'pmedian'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.initial_solver
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.profit_aware_operators
```

````

````{py:attribute} k_prox
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.k_prox
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.k_prox
```

````

````{py:attribute} seed_strategy
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.seed_strategy
:type: str
:value: >
   'lifo'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.seed_strategy
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.params.POPMUSICParams.to_dict
```

````

`````
