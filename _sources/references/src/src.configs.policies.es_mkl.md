# {py:mod}`src.configs.policies.es_mkl`

```{py:module} src.configs.policies.es_mkl
```

```{autodoc2-docstring} src.configs.policies.es_mkl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuKappaLambdaESConfig <src.configs.policies.es_mkl.MuKappaLambdaESConfig>`
  - ```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig
    :summary:
    ```
````

### API

`````{py:class} MuKappaLambdaESConfig
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig
```

````{py:attribute} mu
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.mu
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.mu
```

````

````{py:attribute} kappa
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.kappa
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.kappa
```

````

````{py:attribute} lambda_
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.lambda_
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.lambda_
```

````

````{py:attribute} rho
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.rho
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.rho
```

````

````{py:attribute} recombination_type
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.recombination_type
:type: str
:value: >
   'intermediate'

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.recombination_type
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.n_removal
```

````

````{py:attribute} stagnation_limit
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.stagnation_limit
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.stagnation_limit
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.local_search_iterations
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.es_mkl.MuKappaLambdaESConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mkl.MuKappaLambdaESConfig.post_processing
```

````

`````
