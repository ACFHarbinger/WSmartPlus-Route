# {py:mod}`src.configs.models.decoder`

```{py:module} src.configs.models.decoder
```

```{autodoc2-docstring} src.configs.models.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DecoderConfig <src.configs.models.decoder.DecoderConfig>`
  - ```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig
    :summary:
    ```
````

### API

`````{py:class} DecoderConfig
:canonical: src.configs.models.decoder.DecoderConfig

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig
```

````{py:attribute} type
:canonical: src.configs.models.decoder.DecoderConfig.type
:type: str
:value: >
   'attention'

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.type
```

````

````{py:attribute} embed_dim
:canonical: src.configs.models.decoder.DecoderConfig.embed_dim
:type: int
:value: >
   128

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.embed_dim
```

````

````{py:attribute} hidden_dim
:canonical: src.configs.models.decoder.DecoderConfig.hidden_dim
:type: int
:value: >
   512

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.hidden_dim
```

````

````{py:attribute} n_layers
:canonical: src.configs.models.decoder.DecoderConfig.n_layers
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.n_layers
```

````

````{py:attribute} n_heads
:canonical: src.configs.models.decoder.DecoderConfig.n_heads
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.n_heads
```

````

````{py:attribute} normalization
:canonical: src.configs.models.decoder.DecoderConfig.normalization
:type: src.configs.models.normalization.NormalizationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.normalization
```

````

````{py:attribute} activation
:canonical: src.configs.models.decoder.DecoderConfig.activation
:type: src.configs.models.activation_function.ActivationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.activation
```

````

````{py:attribute} decoding
:canonical: src.configs.models.decoder.DecoderConfig.decoding
:type: src.configs.models.decoding.DecodingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.decoding
```

````

````{py:attribute} dropout
:canonical: src.configs.models.decoder.DecoderConfig.dropout
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.dropout
```

````

````{py:attribute} mask_logits
:canonical: src.configs.models.decoder.DecoderConfig.mask_logits
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.mask_logits
```

````

````{py:attribute} connection_type
:canonical: src.configs.models.decoder.DecoderConfig.connection_type
:type: str
:value: >
   'residual'

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.connection_type
```

````

````{py:attribute} n_predictor_layers
:canonical: src.configs.models.decoder.DecoderConfig.n_predictor_layers
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.n_predictor_layers
```

````

````{py:attribute} tanh_clipping
:canonical: src.configs.models.decoder.DecoderConfig.tanh_clipping
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.tanh_clipping
```

````

````{py:attribute} hyper_expansion
:canonical: src.configs.models.decoder.DecoderConfig.hyper_expansion
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.models.decoder.DecoderConfig.hyper_expansion
```

````

`````
