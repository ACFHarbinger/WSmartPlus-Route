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

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.bpc.BPCConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.profit_aware_operators
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

````{py:attribute} search_strategy
:canonical: src.configs.policies.bpc.BPCConfig.search_strategy
:type: str
:value: >
   'best_first'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.search_strategy
```

````

````{py:attribute} cutting_planes
:canonical: src.configs.policies.bpc.BPCConfig.cutting_planes
:type: str
:value: >
   'rcc'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.cutting_planes
```

````

````{py:attribute} branching_strategy
:canonical: src.configs.policies.bpc.BPCConfig.branching_strategy
:type: str
:value: >
   'ryan_foster'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.branching_strategy
```

````

`````
