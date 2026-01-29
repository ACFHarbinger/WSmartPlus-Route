# {py:mod}`src.configs.model`

```{py:module} src.configs.model
```

```{autodoc2-docstring} src.configs.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelConfig <src.configs.model.ModelConfig>`
  - ```{autodoc2-docstring} src.configs.model.ModelConfig
    :summary:
    ```
````

### API

`````{py:class} ModelConfig
:canonical: src.configs.model.ModelConfig

```{autodoc2-docstring} src.configs.model.ModelConfig
```

````{py:attribute} name
:canonical: src.configs.model.ModelConfig.name
:type: str
:value: >
   'am'

```{autodoc2-docstring} src.configs.model.ModelConfig.name
```

````

````{py:attribute} embed_dim
:canonical: src.configs.model.ModelConfig.embed_dim
:type: int
:value: >
   128

```{autodoc2-docstring} src.configs.model.ModelConfig.embed_dim
```

````

````{py:attribute} hidden_dim
:canonical: src.configs.model.ModelConfig.hidden_dim
:type: int
:value: >
   512

```{autodoc2-docstring} src.configs.model.ModelConfig.hidden_dim
```

````

````{py:attribute} num_encoder_layers
:canonical: src.configs.model.ModelConfig.num_encoder_layers
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.model.ModelConfig.num_encoder_layers
```

````

````{py:attribute} num_decoder_layers
:canonical: src.configs.model.ModelConfig.num_decoder_layers
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.model.ModelConfig.num_decoder_layers
```

````

````{py:attribute} num_heads
:canonical: src.configs.model.ModelConfig.num_heads
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.model.ModelConfig.num_heads
```

````

````{py:attribute} encoder_type
:canonical: src.configs.model.ModelConfig.encoder_type
:type: str
:value: >
   'gat'

```{autodoc2-docstring} src.configs.model.ModelConfig.encoder_type
```

````

````{py:attribute} temporal_horizon
:canonical: src.configs.model.ModelConfig.temporal_horizon
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.model.ModelConfig.temporal_horizon
```

````

````{py:attribute} tanh_clipping
:canonical: src.configs.model.ModelConfig.tanh_clipping
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.model.ModelConfig.tanh_clipping
```

````

````{py:attribute} normalization
:canonical: src.configs.model.ModelConfig.normalization
:type: str
:value: >
   'instance'

```{autodoc2-docstring} src.configs.model.ModelConfig.normalization
```

````

````{py:attribute} activation
:canonical: src.configs.model.ModelConfig.activation
:type: str
:value: >
   'gelu'

```{autodoc2-docstring} src.configs.model.ModelConfig.activation
```

````

````{py:attribute} dropout
:canonical: src.configs.model.ModelConfig.dropout
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.model.ModelConfig.dropout
```

````

````{py:attribute} mask_inner
:canonical: src.configs.model.ModelConfig.mask_inner
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.model.ModelConfig.mask_inner
```

````

````{py:attribute} mask_logits
:canonical: src.configs.model.ModelConfig.mask_logits
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.model.ModelConfig.mask_logits
```

````

````{py:attribute} mask_graph
:canonical: src.configs.model.ModelConfig.mask_graph
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.model.ModelConfig.mask_graph
```

````

````{py:attribute} spatial_bias
:canonical: src.configs.model.ModelConfig.spatial_bias
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.model.ModelConfig.spatial_bias
```

````

````{py:attribute} connection_type
:canonical: src.configs.model.ModelConfig.connection_type
:type: str
:value: >
   'residual'

```{autodoc2-docstring} src.configs.model.ModelConfig.connection_type
```

````

````{py:attribute} num_encoder_sublayers
:canonical: src.configs.model.ModelConfig.num_encoder_sublayers
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.model.ModelConfig.num_encoder_sublayers
```

````

````{py:attribute} num_predictor_layers
:canonical: src.configs.model.ModelConfig.num_predictor_layers
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.model.ModelConfig.num_predictor_layers
```

````

````{py:attribute} learn_affine
:canonical: src.configs.model.ModelConfig.learn_affine
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.model.ModelConfig.learn_affine
```

````

````{py:attribute} track_stats
:canonical: src.configs.model.ModelConfig.track_stats
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.model.ModelConfig.track_stats
```

````

````{py:attribute} epsilon_alpha
:canonical: src.configs.model.ModelConfig.epsilon_alpha
:type: float
:value: >
   1e-05

```{autodoc2-docstring} src.configs.model.ModelConfig.epsilon_alpha
```

````

````{py:attribute} momentum_beta
:canonical: src.configs.model.ModelConfig.momentum_beta
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.model.ModelConfig.momentum_beta
```

````

````{py:attribute} lrnorm_k
:canonical: src.configs.model.ModelConfig.lrnorm_k
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.configs.model.ModelConfig.lrnorm_k
```

````

````{py:attribute} gnorm_groups
:canonical: src.configs.model.ModelConfig.gnorm_groups
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.model.ModelConfig.gnorm_groups
```

````

````{py:attribute} activation_param
:canonical: src.configs.model.ModelConfig.activation_param
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.model.ModelConfig.activation_param
```

````

````{py:attribute} activation_threshold
:canonical: src.configs.model.ModelConfig.activation_threshold
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.configs.model.ModelConfig.activation_threshold
```

````

````{py:attribute} activation_replacement
:canonical: src.configs.model.ModelConfig.activation_replacement
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.configs.model.ModelConfig.activation_replacement
```

````

````{py:attribute} activation_num_parameters
:canonical: src.configs.model.ModelConfig.activation_num_parameters
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.model.ModelConfig.activation_num_parameters
```

````

````{py:attribute} activation_uniform_range
:canonical: src.configs.model.ModelConfig.activation_uniform_range
:type: typing.List[float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.model.ModelConfig.activation_uniform_range
```

````

````{py:attribute} aggregation_graph
:canonical: src.configs.model.ModelConfig.aggregation_graph
:type: str
:value: >
   'avg'

```{autodoc2-docstring} src.configs.model.ModelConfig.aggregation_graph
```

````

````{py:attribute} aggregation_node
:canonical: src.configs.model.ModelConfig.aggregation_node
:type: str
:value: >
   'sum'

```{autodoc2-docstring} src.configs.model.ModelConfig.aggregation_node
```

````

````{py:attribute} spatial_bias_scale
:canonical: src.configs.model.ModelConfig.spatial_bias_scale
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.model.ModelConfig.spatial_bias_scale
```

````

````{py:attribute} hyper_expansion
:canonical: src.configs.model.ModelConfig.hyper_expansion
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.model.ModelConfig.hyper_expansion
```

````

````{py:attribute} policy_config
:canonical: src.configs.model.ModelConfig.policy_config
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.model.ModelConfig.policy_config
```

````

`````
