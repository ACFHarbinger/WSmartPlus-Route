# {py:mod}`src.configs.policies.alns`

```{py:module} src.configs.policies.alns
```

```{autodoc2-docstring} src.configs.policies.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSConfig <src.configs.policies.alns.ALNSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig
    :summary:
    ```
````

### API

`````{py:class} ALNSConfig
:canonical: src.configs.policies.alns.ALNSConfig

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.alns.ALNSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.alns.ALNSConfig.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.configs.policies.alns.ALNSConfig.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.alns.ALNSConfig.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.cooling_rate
```

````

````{py:attribute} reaction_factor
:canonical: src.configs.policies.alns.ALNSConfig.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.reaction_factor
```

````

````{py:attribute} min_removal
:canonical: src.configs.policies.alns.ALNSConfig.min_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.min_removal
```

````

````{py:attribute} max_removal_pct
:canonical: src.configs.policies.alns.ALNSConfig.max_removal_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.max_removal_pct
```

````

````{py:attribute} engine
:canonical: src.configs.policies.alns.ALNSConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.alns.ALNSConfig.must_go
:type: typing.Optional[typing.List[src.configs.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.alns.ALNSConfig.post_processing
:type: typing.Optional[typing.List[src.configs.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.post_processing
```

````

`````
