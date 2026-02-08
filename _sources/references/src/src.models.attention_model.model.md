# {py:mod}`src.models.attention_model.model`

```{py:module} src.models.attention_model.model
```

```{autodoc2-docstring} src.models.attention_model.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionModel <src.models.attention_model.model.AttentionModel>`
  - ```{autodoc2-docstring} src.models.attention_model.model.AttentionModel
    :summary:
    ```
````

### API

`````{py:class} AttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, component_factory: logic.src.models.subnets.factories.NeuralComponentFactory, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, aggregation: str = 'sum', aggregation_graph: str = 'avg', tanh_clipping: float = TANH_CLIPPING, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, n_heads: int = 8, checkpoint_encoder: bool = False, shrink_size: typing.Optional[int] = None, pomo_size: int = 0, temporal_horizon: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, entropy_weight: float = 0.0, predictor_layers: typing.Optional[int] = None, connection_type: str = 'residual', hyper_expansion: int = FEED_FORWARD_EXPANSION, decoder_type: str = 'attention', **kwargs)
:canonical: src.models.attention_model.model.AttentionModel

Bases: {py:obj}`src.models.attention_model.decoding.DecodingMixin`, {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel.__init__
```

````{py:method} _init_parameters(embed_dim: int, hidden_dim: int, problem: typing.Any, n_heads: int, pomo_size: int, checkpoint_encoder: bool, aggregation_graph: str, temporal_horizon: int, tanh_clipping: float)
:canonical: src.models.attention_model.model.AttentionModel._init_parameters

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel._init_parameters
```

````

````{py:method} _init_context_embedder(temporal_horizon: int)
:canonical: src.models.attention_model.model.AttentionModel._init_context_embedder

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel._init_context_embedder
```

````

````{py:property} is_vrpp
:canonical: src.models.attention_model.model.AttentionModel.is_vrpp

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel.is_vrpp
```

````

````{py:property} is_wc
:canonical: src.models.attention_model.model.AttentionModel.is_wc

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel.is_wc
```

````

````{py:method} _init_components(component_factory: logic.src.models.subnets.factories.NeuralComponentFactory, step_context_dim: int, n_encode_layers: int, n_encode_sublayers: typing.Optional[int], n_decode_layers: typing.Optional[int], norm_config: logic.src.configs.models.normalization.NormalizationConfig, activation_config: logic.src.configs.models.activation_function.ActivationConfig, dropout_rate: float, aggregation: str, hyper_expansion: int, connection_type: str, predictor_layers: typing.Optional[int], tanh_clipping: float, mask_inner: bool, mask_logits: bool, mask_graph: bool, shrink_size: typing.Optional[int], spatial_bias: bool, spatial_bias_scale: float, decoder_type: str = 'attention')
:canonical: src.models.attention_model.model.AttentionModel._init_components

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel._init_components
```

````

````{py:method} _get_initial_embeddings(input: typing.Dict[str, torch.Tensor])
:canonical: src.models.attention_model.model.AttentionModel._get_initial_embeddings

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel._get_initial_embeddings
```

````

````{py:method} forward(input: typing.Dict[str, torch.Tensor], env: typing.Optional[typing.Any] = None, strategy: typing.Optional[str] = None, return_pi: bool = False, pad: bool = False, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.attention_model.model.AttentionModel.forward

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel.forward
```

````

````{py:method} precompute_fixed(input: typing.Dict[str, torch.Tensor], edges: typing.Optional[torch.Tensor])
:canonical: src.models.attention_model.model.AttentionModel.precompute_fixed

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel.precompute_fixed
```

````

````{py:method} expand(t)
:canonical: src.models.attention_model.model.AttentionModel.expand

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel.expand
```

````

`````
