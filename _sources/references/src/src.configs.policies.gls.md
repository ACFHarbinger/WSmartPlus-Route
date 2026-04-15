# {py:mod}`src.configs.policies.gls`

```{py:module} src.configs.policies.gls
```

```{autodoc2-docstring} src.configs.policies.gls
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GLSConfig <src.configs.policies.gls.GLSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.gls.GLSConfig
    :summary:
    ```
````

### API

`````{py:class} GLSConfig
:canonical: src.configs.policies.gls.GLSConfig

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig
```

````{py:attribute} lambda_param
:canonical: src.configs.policies.gls.GLSConfig.lambda_param
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.lambda_param
```

````

````{py:attribute} alpha_param
:canonical: src.configs.policies.gls.GLSConfig.alpha_param
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.alpha_param
```

````

````{py:attribute} penalty_cycles
:canonical: src.configs.policies.gls.GLSConfig.penalty_cycles
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.penalty_cycles
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.gls.GLSConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.gls.GLSConfig.n_llh
:type: int
:value: >
   6

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.n_llh
```

````

````{py:attribute} inner_iterations
:canonical: src.configs.policies.gls.GLSConfig.inner_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.inner_iterations
```

````

````{py:attribute} fls_coupling_prob
:canonical: src.configs.policies.gls.GLSConfig.fls_coupling_prob
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.fls_coupling_prob
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.gls.GLSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.gls.GLSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.gls.GLSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.gls.GLSConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.gls.GLSConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.gls.GLSConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.post_processing
```

````

`````
