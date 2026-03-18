# {py:mod}`src.configs.policies.bpc`

```{py:module} src.configs.policies.bpc
```

```{autodoc2-docstring} src.configs.policies.bpc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BPCConfig <src.configs.policies.bpc.BPCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig
    :summary:
    ```
````

### API

`````{py:class} BPCConfig
:canonical: src.configs.policies.bpc.BPCConfig

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.bpc.BPCConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.time_limit
```

````

````{py:attribute} engine
:canonical: src.configs.policies.bpc.BPCConfig.engine
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.engine
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.bpc.BPCConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.bpc.BPCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.bpc.BPCConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.bpc.BPCConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.post_processing
```

````

`````
