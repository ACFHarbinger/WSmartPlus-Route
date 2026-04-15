# {py:mod}`src.configs.policies.aco_ks`

```{py:module} src.configs.policies.aco_ks
```

```{autodoc2-docstring} src.configs.policies.aco_ks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KSparseACOConfig <src.configs.policies.aco_ks.KSparseACOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig
    :summary:
    ```
````

### API

`````{py:class} KSparseACOConfig
:canonical: src.configs.policies.aco_ks.KSparseACOConfig

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig
```

````{py:attribute} n_ants
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.n_ants
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.n_ants
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.alpha
```

````

````{py:attribute} beta
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.beta
```

````

````{py:attribute} rho
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.rho
```

````

````{py:attribute} tau_0
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.tau_0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.tau_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.seed
```

````

````{py:attribute} q0
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.q0
```

````

````{py:attribute} k_sparse
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.k_sparse
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.k_sparse
```

````

````{py:attribute} local_search
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.local_search
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.local_search
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.local_search_iterations
```

````

````{py:attribute} elitist_weight
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.elitist_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.elitist_weight
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.aco_ks.KSparseACOConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco_ks.KSparseACOConfig.post_processing
```

````

`````
