# {py:mod}`src.configs.policies.es_mcl`

```{py:module} src.configs.policies.es_mcl
```

```{autodoc2-docstring} src.configs.policies.es_mcl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuCommaLambdaESConfig <src.configs.policies.es_mcl.MuCommaLambdaESConfig>`
  - ```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig
    :summary:
    ```
````

### API

`````{py:class} MuCommaLambdaESConfig
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig
```

````{py:attribute} mu
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.mu
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.mu
```

````

````{py:attribute} lambda_
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.lambda_
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.lambda_
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.n_removal
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.es_mcl.MuCommaLambdaESConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mcl.MuCommaLambdaESConfig.post_processing
```

````

`````
