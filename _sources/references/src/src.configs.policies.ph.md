# {py:mod}`src.configs.policies.ph`

```{py:module} src.configs.policies.ph
```

```{autodoc2-docstring} src.configs.policies.ph
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PHConfig <src.configs.policies.ph.PHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ph.PHConfig
    :summary:
    ```
````

### API

`````{py:class} PHConfig
:canonical: src.configs.policies.ph.PHConfig

```{autodoc2-docstring} src.configs.policies.ph.PHConfig
```

````{py:attribute} rho
:canonical: src.configs.policies.ph.PHConfig.rho
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.rho
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ph.PHConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.max_iterations
```

````

````{py:attribute} convergence_tol
:canonical: src.configs.policies.ph.PHConfig.convergence_tol
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.convergence_tol
```

````

````{py:attribute} sub_solver
:canonical: src.configs.policies.ph.PHConfig.sub_solver
:type: str
:value: >
   'bc'

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.sub_solver
```

````

````{py:attribute} num_scenarios
:canonical: src.configs.policies.ph.PHConfig.num_scenarios
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.num_scenarios
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ph.PHConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.time_limit
```

````

````{py:attribute} verbose
:canonical: src.configs.policies.ph.PHConfig.verbose
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.verbose
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ph.PHConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ph.PHConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.post_processing
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ph.PHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.seed
```

````

`````
