# {py:mod}`src.models.attention_model.setup`

```{py:module} src.models.attention_model.setup
```

```{autodoc2-docstring} src.models.attention_model.setup
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SetupMixin <src.models.attention_model.setup.SetupMixin>`
  - ```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin
    :summary:
    ```
````

### API

`````{py:class} SetupMixin()
:canonical: src.models.attention_model.setup.SetupMixin

```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin.__init__
```

````{py:method} _init_parameters(embed_dim: int, hidden_dim: int, problem: typing.Any, n_heads: int, pomo_size: int, checkpoint_encoder: bool, aggregation_graph: str, temporal_horizon: int, tanh_clipping: float)
:canonical: src.models.attention_model.setup.SetupMixin._init_parameters

```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin._init_parameters
```

````

````{py:method} _init_context_embedder(temporal_horizon: int)
:canonical: src.models.attention_model.setup.SetupMixin._init_context_embedder

```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin._init_context_embedder
```

````

````{py:property} is_vrpp
:canonical: src.models.attention_model.setup.SetupMixin.is_vrpp

```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin.is_vrpp
```

````

````{py:property} is_wc
:canonical: src.models.attention_model.setup.SetupMixin.is_wc

```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin.is_wc
```

````

````{py:method} _init_components(component_factory: logic.src.models.subnets.factories.NeuralComponentFactory, step_context_dim: int, n_encode_layers: int, n_encode_sublayers: typing.Optional[int], n_decode_layers: typing.Optional[int], normalization: str, norm_learn_affine: bool, norm_track_stats: bool, norm_eps_alpha: float, norm_momentum_beta: float, lrnorm_k: float, gnorm_groups: int, activation_function: str, af_param: float, af_threshold: float, af_replacement_value: float, af_num_params: int, af_uniform_range: typing.List[float], dropout_rate: float, aggregation: str, hyper_expansion: int, connection_type: str, predictor_layers: typing.Optional[int], tanh_clipping: float, mask_inner: bool, mask_logits: bool, mask_graph: bool, shrink_size: typing.Optional[int], spatial_bias: bool, spatial_bias_scale: float, decoder_type: str = 'attention')
:canonical: src.models.attention_model.setup.SetupMixin._init_components

```{autodoc2-docstring} src.models.attention_model.setup.SetupMixin._init_components
```

````

`````
