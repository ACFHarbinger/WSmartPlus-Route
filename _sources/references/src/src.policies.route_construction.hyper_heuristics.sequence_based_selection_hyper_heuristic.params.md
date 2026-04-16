# {py:mod}`src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params`

```{py:module} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SSHHParams <src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams
    :summary:
    ```
````

### API

`````{py:class} SSHHParams
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams
```

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.time_limit
```

````

````{py:attribute} threshold_infeasible
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.threshold_infeasible
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.threshold_infeasible
```

````

````{py:attribute} threshold_feasible_base
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.threshold_feasible_base
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.threshold_feasible_base
```

````

````{py:attribute} threshold_decay_rate
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.threshold_decay_rate
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.threshold_decay_rate
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.params.SSHHParams.from_config
```

````

`````
