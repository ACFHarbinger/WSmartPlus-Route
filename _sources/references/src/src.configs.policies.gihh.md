# {py:mod}`src.configs.policies.gihh`

```{py:module} src.configs.policies.gihh
```

```{autodoc2-docstring} src.configs.policies.gihh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GIHHConfig <src.configs.policies.gihh.GIHHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig
    :summary:
    ```
````

### API

`````{py:class} GIHHConfig
:canonical: src.configs.policies.gihh.GIHHConfig

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.gihh.GIHHConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.gihh.GIHHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.seed
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.gihh.GIHHConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.max_iterations
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.gihh.GIHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.gihh.GIHHConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.profit_aware_operators
```

````

````{py:attribute} seg
:canonical: src.configs.policies.gihh.GIHHConfig.seg
:type: int
:value: >
   80

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.seg
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.gihh.GIHHConfig.alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.alpha
```

````

````{py:attribute} beta
:canonical: src.configs.policies.gihh.GIHHConfig.beta
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.beta
```

````

````{py:attribute} gamma
:canonical: src.configs.policies.gihh.GIHHConfig.gamma
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.gamma
```

````

````{py:attribute} min_prob
:canonical: src.configs.policies.gihh.GIHHConfig.min_prob
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.min_prob
```

````

````{py:attribute} nonimp_threshold
:canonical: src.configs.policies.gihh.GIHHConfig.nonimp_threshold
:type: int
:value: >
   150

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.nonimp_threshold
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.gihh.GIHHConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.gihh.GIHHConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.post_processing
```

````

`````
