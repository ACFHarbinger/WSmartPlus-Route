# {py:mod}`src.configs.policies.sans`

```{py:module} src.configs.policies.sans
```

```{autodoc2-docstring} src.configs.policies.sans
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SANSConfig <src.configs.policies.sans.SANSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.sans.SANSConfig
    :summary:
    ```
````

### API

`````{py:class} SANSConfig
:canonical: src.configs.policies.sans.SANSConfig

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.sans.SANSConfig.engine
:type: typing.Literal[new, og]
:value: >
   'new'

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.engine
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.sans.SANSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.time_limit
```

````

````{py:attribute} perc_bins_can_overflow
:canonical: src.configs.policies.sans.SANSConfig.perc_bins_can_overflow
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.perc_bins_can_overflow
```

````

````{py:attribute} T_min
:canonical: src.configs.policies.sans.SANSConfig.T_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.T_min
```

````

````{py:attribute} T_init
:canonical: src.configs.policies.sans.SANSConfig.T_init
:type: float
:value: >
   75.0

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.T_init
```

````

````{py:attribute} iterations_per_T
:canonical: src.configs.policies.sans.SANSConfig.iterations_per_T
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.iterations_per_T
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.sans.SANSConfig.alpha
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.alpha
```

````

````{py:attribute} combination
:canonical: src.configs.policies.sans.SANSConfig.combination
:type: typing.Optional[typing.Literal[a, b]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.combination
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.sans.SANSConfig.must_go
:type: typing.Optional[typing.List[src.configs.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.sans.SANSConfig.post_processing
:type: typing.Optional[typing.List[src.configs.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sans.SANSConfig.post_processing
```

````

`````
