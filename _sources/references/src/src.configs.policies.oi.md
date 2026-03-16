# {py:mod}`src.configs.policies.oi`

```{py:module} src.configs.policies.oi
```

```{autodoc2-docstring} src.configs.policies.oi
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OIConfig <src.configs.policies.oi.OIConfig>`
  - ```{autodoc2-docstring} src.configs.policies.oi.OIConfig
    :summary:
    ```
````

### API

`````{py:class} OIConfig
:canonical: src.configs.policies.oi.OIConfig

```{autodoc2-docstring} src.configs.policies.oi.OIConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.oi.OIConfig.engine
:type: str
:value: >
   'oi'

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.oi.OIConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.oi.OIConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.oi.OIConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.oi.OIConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.oi.OIConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.oi.OIConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.oi.OIConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.oi.OIConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.oi.OIConfig.post_processing
```

````

`````
