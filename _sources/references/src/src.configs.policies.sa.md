# {py:mod}`src.configs.policies.sa`

```{py:module} src.configs.policies.sa
```

```{autodoc2-docstring} src.configs.policies.sa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAConfig <src.configs.policies.sa.SAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.sa.SAConfig
    :summary:
    ```
````

### API

`````{py:class} SAConfig
:canonical: src.configs.policies.sa.SAConfig

```{autodoc2-docstring} src.configs.policies.sa.SAConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.sa.SAConfig.engine
:type: str
:value: >
   'sa'

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.engine
```

````

````{py:attribute} initial_temp
:canonical: src.configs.policies.sa.SAConfig.initial_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.initial_temp
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.sa.SAConfig.alpha
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.alpha
```

````

````{py:attribute} min_temp
:canonical: src.configs.policies.sa.SAConfig.min_temp
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.min_temp
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.sa.SAConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.sa.SAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.sa.SAConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.sa.SAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.sa.SAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.sa.SAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.sa.SAConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.sa.SAConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.post_processing
```

````

`````
