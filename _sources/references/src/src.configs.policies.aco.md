# {py:mod}`src.configs.policies.aco`

```{py:module} src.configs.policies.aco
```

```{autodoc2-docstring} src.configs.policies.aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACOConfig <src.configs.policies.aco.ACOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.aco.ACOConfig
    :summary:
    ```
````

### API

`````{py:class} ACOConfig
:canonical: src.configs.policies.aco.ACOConfig

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig
```

````{py:attribute} n_ants
:canonical: src.configs.policies.aco.ACOConfig.n_ants
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.n_ants
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.aco.ACOConfig.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.alpha
```

````

````{py:attribute} beta
:canonical: src.configs.policies.aco.ACOConfig.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.beta
```

````

````{py:attribute} rho
:canonical: src.configs.policies.aco.ACOConfig.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.rho
```

````

````{py:attribute} tau_0
:canonical: src.configs.policies.aco.ACOConfig.tau_0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.configs.policies.aco.ACOConfig.tau_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.configs.policies.aco.ACOConfig.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.aco.ACOConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.aco.ACOConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.time_limit
```

````

````{py:attribute} q0
:canonical: src.configs.policies.aco.ACOConfig.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.q0
```

````

````{py:attribute} k_sparse
:canonical: src.configs.policies.aco.ACOConfig.k_sparse
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.k_sparse
```

````

````{py:attribute} sequence_length
:canonical: src.configs.policies.aco.ACOConfig.sequence_length
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.sequence_length
```

````

````{py:attribute} local_search
:canonical: src.configs.policies.aco.ACOConfig.local_search
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.local_search
```

````

````{py:attribute} elitist_weight
:canonical: src.configs.policies.aco.ACOConfig.elitist_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.elitist_weight
```

````

````{py:attribute} operators
:canonical: src.configs.policies.aco.ACOConfig.operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.operators
```

````

````{py:attribute} engine
:canonical: src.configs.policies.aco.ACOConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.aco.ACOConfig.must_go
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.aco.ACOConfig.post_processing
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco.ACOConfig.post_processing
```

````

`````
