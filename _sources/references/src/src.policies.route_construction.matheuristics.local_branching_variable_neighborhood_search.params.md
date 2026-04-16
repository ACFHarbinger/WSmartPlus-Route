# {py:mod}`src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params`

```{py:module} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LBVNSParams <src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams
    :summary:
    ```
````

### API

`````{py:class} LBVNSParams
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams
```

````{py:attribute} k_min
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.k_min
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.k_min
```

````

````{py:attribute} k_max
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.k_max
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.k_max
```

````

````{py:attribute} k_step
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.k_step
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.k_step
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.time_limit
```

````

````{py:attribute} time_limit_per_lb
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.time_limit_per_lb
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.time_limit_per_lb
```

````

````{py:attribute} max_lb_iterations
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.max_lb_iterations
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.max_lb_iterations
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.mip_gap
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.seed
```

````

````{py:attribute} engine
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.engine
```

````

````{py:attribute} framework
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.framework
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.acceptance_criterion
:type: logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.acceptance_criterion
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams.to_dict
```

````

`````
