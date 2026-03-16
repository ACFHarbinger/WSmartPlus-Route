# {py:mod}`src.configs.policies.es_mpl`

```{py:module} src.configs.policies.es_mpl
```

```{autodoc2-docstring} src.configs.policies.es_mpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuPlusLambdaESConfig <src.configs.policies.es_mpl.MuPlusLambdaESConfig>`
  - ```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig
    :summary:
    ```
````

### API

`````{py:class} MuPlusLambdaESConfig
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig
```

````{py:attribute} mu
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.mu
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.mu
```

````

````{py:attribute} lambda_
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.lambda_
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.lambda_
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.n_removal
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.es_mpl.MuPlusLambdaESConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.es_mpl.MuPlusLambdaESConfig.post_processing
```

````

`````
