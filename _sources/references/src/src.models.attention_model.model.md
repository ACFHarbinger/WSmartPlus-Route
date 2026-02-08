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

````{py:class} AttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, component_factory: logic.src.models.subnets.factories.NeuralComponentFactory, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, aggregation: str = 'sum', aggregation_graph: str = 'avg', tanh_clipping: float = TANH_CLIPPING, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, normalization: str = 'batch', norm_learn_affine: bool = True, norm_track_stats: bool = False, norm_eps_alpha: float = NORM_EPSILON, norm_momentum_beta: float = 0.1, lrnorm_k: float = 1.0, gnorm_groups: int = 3, activation_function: str = 'gelu', af_param: float = 1.0, af_threshold: float = 6.0, af_replacement_value: float = 6.0, af_num_params: int = 3, af_uniform_range: typing.List[float] = [0.125, 1 / 3], n_heads: int = 8, checkpoint_encoder: bool = False, shrink_size: typing.Optional[int] = None, pomo_size: int = 0, temporal_horizon: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, entropy_weight: float = 0.0, predictor_layers: typing.Optional[int] = None, connection_type: str = 'residual', hyper_expansion: int = FEED_FORWARD_EXPANSION, decoder_type: str = 'attention')
:canonical: src.models.attention_model.model.AttentionModel

Bases: {py:obj}`src.models.attention_model.setup.SetupMixin`, {py:obj}`src.models.attention_model.forward.ForwardMixin`, {py:obj}`src.models.attention_model.decoding.DecodingMixin`, {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.model.AttentionModel.__init__
```

````
