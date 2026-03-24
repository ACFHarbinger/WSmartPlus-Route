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

````{py:attribute} engine
:canonical: src.configs.policies.gls.GLSConfig.engine
:type: str
:value: >
   'gls'

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.engine
```

````

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

````{py:attribute} max_restarts
:canonical: src.configs.policies.gls.GLSConfig.max_restarts
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.max_restarts
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
   5

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.n_llh
```

````

````{py:attribute} inner_iterations
:canonical: src.configs.policies.gls.GLSConfig.inner_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.gls.GLSConfig.inner_iterations
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
