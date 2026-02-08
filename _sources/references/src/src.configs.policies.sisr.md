# {py:mod}`src.configs.policies.sisr`

```{py:module} src.configs.policies.sisr
```

```{autodoc2-docstring} src.configs.policies.sisr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SISRConfig <src.configs.policies.sisr.SISRConfig>`
  - ```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig
    :summary:
    ```
````

### API

`````{py:class} SISRConfig
:canonical: src.configs.policies.sisr.SISRConfig

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.sisr.SISRConfig.time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.sisr.SISRConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.configs.policies.sisr.SISRConfig.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.sisr.SISRConfig.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.cooling_rate
```

````

````{py:attribute} max_string_len
:canonical: src.configs.policies.sisr.SISRConfig.max_string_len
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.max_string_len
```

````

````{py:attribute} avg_string_len
:canonical: src.configs.policies.sisr.SISRConfig.avg_string_len
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.avg_string_len
```

````

````{py:attribute} blink_rate
:canonical: src.configs.policies.sisr.SISRConfig.blink_rate
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.blink_rate
```

````

````{py:attribute} destroy_ratio
:canonical: src.configs.policies.sisr.SISRConfig.destroy_ratio
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.destroy_ratio
```

````

````{py:attribute} engine
:canonical: src.configs.policies.sisr.SISRConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.sisr.SISRConfig.must_go
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.sisr.SISRConfig.post_processing
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sisr.SISRConfig.post_processing
```

````

`````
