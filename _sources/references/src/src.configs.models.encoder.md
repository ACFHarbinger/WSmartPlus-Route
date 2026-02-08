# {py:mod}`src.configs.models.encoder`

```{py:module} src.configs.models.encoder
```

```{autodoc2-docstring} src.configs.models.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EncoderConfig <src.configs.models.encoder.EncoderConfig>`
  - ```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig
    :summary:
    ```
````

### API

`````{py:class} EncoderConfig
:canonical: src.configs.models.encoder.EncoderConfig

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig
```

````{py:attribute} type
:canonical: src.configs.models.encoder.EncoderConfig.type
:type: str
:value: >
   'gat'

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.type
```

````

````{py:attribute} embed_dim
:canonical: src.configs.models.encoder.EncoderConfig.embed_dim
:type: int
:value: >
   128

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.embed_dim
```

````

````{py:attribute} hidden_dim
:canonical: src.configs.models.encoder.EncoderConfig.hidden_dim
:type: int
:value: >
   512

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.hidden_dim
```

````

````{py:attribute} n_layers
:canonical: src.configs.models.encoder.EncoderConfig.n_layers
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.n_layers
```

````

````{py:attribute} n_heads
:canonical: src.configs.models.encoder.EncoderConfig.n_heads
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.n_heads
```

````

````{py:attribute} n_sublayers
:canonical: src.configs.models.encoder.EncoderConfig.n_sublayers
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.n_sublayers
```

````

````{py:attribute} normalization
:canonical: src.configs.models.encoder.EncoderConfig.normalization
:type: src.configs.models.normalization.NormalizationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.normalization
```

````

````{py:attribute} activation
:canonical: src.configs.models.encoder.EncoderConfig.activation
:type: src.configs.models.activation_function.ActivationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.activation
```

````

````{py:attribute} dropout
:canonical: src.configs.models.encoder.EncoderConfig.dropout
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.dropout
```

````

````{py:attribute} mask_inner
:canonical: src.configs.models.encoder.EncoderConfig.mask_inner
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.mask_inner
```

````

````{py:attribute} mask_graph
:canonical: src.configs.models.encoder.EncoderConfig.mask_graph
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.mask_graph
```

````

````{py:attribute} spatial_bias
:canonical: src.configs.models.encoder.EncoderConfig.spatial_bias
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.spatial_bias
```

````

````{py:attribute} connection_type
:canonical: src.configs.models.encoder.EncoderConfig.connection_type
:type: str
:value: >
   'residual'

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.connection_type
```

````

````{py:attribute} aggregation_graph
:canonical: src.configs.models.encoder.EncoderConfig.aggregation_graph
:type: str
:value: >
   'avg'

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.aggregation_graph
```

````

````{py:attribute} aggregation_node
:canonical: src.configs.models.encoder.EncoderConfig.aggregation_node
:type: str
:value: >
   'sum'

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.aggregation_node
```

````

````{py:attribute} spatial_bias_scale
:canonical: src.configs.models.encoder.EncoderConfig.spatial_bias_scale
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.spatial_bias_scale
```

````

````{py:attribute} hyper_expansion
:canonical: src.configs.models.encoder.EncoderConfig.hyper_expansion
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.models.encoder.EncoderConfig.hyper_expansion
```

````

`````
