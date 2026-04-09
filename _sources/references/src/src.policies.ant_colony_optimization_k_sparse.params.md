# {py:mod}`src.policies.ant_colony_optimization_k_sparse.params`

```{py:module} src.policies.ant_colony_optimization_k_sparse.params
```

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KSACOParams <src.policies.ant_colony_optimization_k_sparse.params.KSACOParams>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams
    :summary:
    ```
````

### API

`````{py:class} KSACOParams
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams
```

````{py:attribute} n_ants
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.n_ants
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.n_ants
```

````

````{py:attribute} k_sparse
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.k_sparse
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.k_sparse
```

````

````{py:attribute} alpha
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.beta
```

````

````{py:attribute} rho
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.rho
```

````

````{py:attribute} scale
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.scale
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.scale
```

````

````{py:attribute} tau_0
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.tau_0
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.tau_min
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.time_limit
```

````

````{py:attribute} local_search
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.local_search
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.local_search
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.local_search_iterations
```

````

````{py:attribute} elitist_weight
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.elitist_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.elitist_weight
```

````

````{py:attribute} vrpp
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.profit_aware_operators
```

````

````{py:attribute} q0
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.q0
```

````

````{py:attribute} seed
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.seed
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.__post_init__

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.__post_init__
```

````

````{py:method} from_config(config: typing.Union[logic.src.configs.policies.KSparseACOConfig, typing.Dict[str, typing.Any]]) -> src.policies.ant_colony_optimization_k_sparse.params.KSACOParams
:canonical: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.ant_colony_optimization_k_sparse.params.KSACOParams.from_config
```

````

`````
