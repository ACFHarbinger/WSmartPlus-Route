# {py:mod}`src.configs.decoding`

```{py:module} src.configs.decoding
```

```{autodoc2-docstring} src.configs.decoding
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DecodingConfig <src.configs.decoding.DecodingConfig>`
  - ```{autodoc2-docstring} src.configs.decoding.DecodingConfig
    :summary:
    ```
````

### API

`````{py:class} DecodingConfig
:canonical: src.configs.decoding.DecodingConfig

```{autodoc2-docstring} src.configs.decoding.DecodingConfig
```

````{py:attribute} temperature
:canonical: src.configs.decoding.DecodingConfig.temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.temperature
```

````

````{py:attribute} top_k
:canonical: src.configs.decoding.DecodingConfig.top_k
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.top_k
```

````

````{py:attribute} top_p
:canonical: src.configs.decoding.DecodingConfig.top_p
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.top_p
```

````

````{py:attribute} tanh_clipping
:canonical: src.configs.decoding.DecodingConfig.tanh_clipping
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.tanh_clipping
```

````

````{py:attribute} mask_logits
:canonical: src.configs.decoding.DecodingConfig.mask_logits
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.mask_logits
```

````

````{py:attribute} multistart
:canonical: src.configs.decoding.DecodingConfig.multistart
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.multistart
```

````

````{py:attribute} num_starts
:canonical: src.configs.decoding.DecodingConfig.num_starts
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.num_starts
```

````

````{py:attribute} select_best
:canonical: src.configs.decoding.DecodingConfig.select_best
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.decoding.DecodingConfig.select_best
```

````

`````
