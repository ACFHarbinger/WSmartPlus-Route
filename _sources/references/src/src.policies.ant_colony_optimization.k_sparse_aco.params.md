# {py:mod}`src.policies.ant_colony_optimization.k_sparse_aco.params`

```{py:module} src.policies.ant_colony_optimization.k_sparse_aco.params
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACOParams <src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams
    :summary:
    ```
````

### API

`````{py:class} ACOParams
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams
```

````{py:attribute} n_ants
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.n_ants
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.n_ants
```

````

````{py:attribute} k_sparse
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.k_sparse
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.k_sparse
```

````

````{py:attribute} alpha
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.beta
```

````

````{py:attribute} rho
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.rho
```

````

````{py:attribute} q0
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.q0
```

````

````{py:attribute} tau_0
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.tau_0
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.tau_min
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.time_limit
```

````

````{py:attribute} local_search
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.local_search
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.local_search
```

````

````{py:attribute} elitist_weight
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.elitist_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.elitist_weight
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.__post_init__

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.__post_init__
```

````

````{py:method} from_config(config: logic.src.configs.policies.ACOConfig) -> src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams.from_config
```

````

`````
