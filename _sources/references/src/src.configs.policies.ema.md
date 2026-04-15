# {py:mod}`src.configs.policies.ema`

```{py:module} src.configs.policies.ema
```

```{autodoc2-docstring} src.configs.policies.ema
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EMAConfig <src.configs.policies.ema.EMAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ema.EMAConfig
    :summary:
    ```
````

### API

`````{py:class} EMAConfig
:canonical: src.configs.policies.ema.EMAConfig

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig
```

````{py:attribute} max_iterations
:canonical: src.configs.policies.ema.EMAConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.max_iterations
```

````

````{py:attribute} rule
:canonical: src.configs.policies.ema.EMAConfig.rule
:type: str
:value: >
   'G-VOT'

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.rule
```

````

````{py:attribute} criteria
:canonical: src.configs.policies.ema.EMAConfig.criteria
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.criteria
```

````

````{py:attribute} sub_params
:canonical: src.configs.policies.ema.EMAConfig.sub_params
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.sub_params
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ema.EMAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.ema.EMAConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ema.EMAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ema.EMAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ema.EMAConfig.profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ema.EMAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ema.EMAConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ema.EMAConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ema.EMAConfig.post_processing
```

````

`````
