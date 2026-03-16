# {py:mod}`src.configs.policies.bb`

```{py:module} src.configs.policies.bb
```

```{autodoc2-docstring} src.configs.policies.bb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BBConfig <src.configs.policies.bb.BBConfig>`
  - ```{autodoc2-docstring} src.configs.policies.bb.BBConfig
    :summary:
    ```
````

### API

`````{py:class} BBConfig
:canonical: src.configs.policies.bb.BBConfig

```{autodoc2-docstring} src.configs.policies.bb.BBConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.bb.BBConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.bb.BBConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.mip_gap
```

````

````{py:attribute} branching_strategy
:canonical: src.configs.policies.bb.BBConfig.branching_strategy
:type: str
:value: >
   'most_fractional'

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.branching_strategy
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.bb.BBConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.bb.BBConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.bb.BBConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.bb.BBConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.post_processing
```

````

`````
