# {py:mod}`src.policies.ant_colony_optimization.hyper_heuristic_aco.params`

```{py:module} src.policies.ant_colony_optimization.hyper_heuristic_aco.params
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperACOParams <src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams
    :summary:
    ```
````

### API

`````{py:class} HyperACOParams
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams
```

````{py:attribute} n_ants
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.n_ants
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.n_ants
```

````

````{py:attribute} alpha
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.beta
```

````

````{py:attribute} rho
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.rho
```

````

````{py:attribute} tau_0
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.tau_0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.tau_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.time_limit
```

````

````{py:attribute} sequence_length
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.sequence_length
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.sequence_length
```

````

````{py:attribute} q0
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.q0
```

````

````{py:attribute} operators
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.operators
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.__post_init__

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.__post_init__
```

````

````{py:method} from_config(config: logic.src.configs.policies.ACOConfig) -> src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams.from_config
```

````

`````
