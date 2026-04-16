# {py:mod}`src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params`

```{py:module} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`QDEParams <src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams
    :summary:
    ```
````

### API

`````{py:class} QDEParams
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams
```

````{py:attribute} pop_size
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.pop_size
```

````

````{py:attribute} F
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.F
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.F
```

````

````{py:attribute} CR
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.CR
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.CR
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.time_limit
```

````

````{py:attribute} delta_theta
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.delta_theta
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.delta_theta
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.local_search_iterations
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams.from_config
```

````

`````
