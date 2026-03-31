# {py:mod}`src.policies.ant_colony_optimization_hyper_heuristic.params`

```{py:module} src.policies.ant_colony_optimization_hyper_heuristic.params
```

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperACOParams <src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams
    :summary:
    ```
````

### API

`````{py:class} HyperACOParams
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams
```

````{py:attribute} n_ants
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.n_ants
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.n_ants
```

````

````{py:attribute} alpha
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.beta
```

````

````{py:attribute} rho
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.rho
```

````

````{py:attribute} tau_0
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.tau_0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.tau_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.time_limit
```

````

````{py:attribute} q0
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.q0
```

````

````{py:attribute} lambda_factor
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.lambda_factor
:type: float
:value: >
   1.0001

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.lambda_factor
```

````

````{py:attribute} operators
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.operators
```

````

````{py:attribute} vrpp
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.seed
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.__post_init__

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams
:canonical: src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.ant_colony_optimization_hyper_heuristic.params.HyperACOParams.from_config
```

````

`````
