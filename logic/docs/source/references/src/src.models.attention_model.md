# {py:mod}`src.models.attention_model`

```{py:module} src.models.attention_model
```

```{autodoc2-docstring} src.models.attention_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionModel <src.models.attention_model.AttentionModel>`
  - ```{autodoc2-docstring} src.models.attention_model.AttentionModel
    :summary:
    ```
````

### API

`````{py:class} AttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, component_factory: logic.src.models.model_factory.NeuralComponentFactory, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, aggregation: str = 'sum', aggregation_graph: str = 'avg', tanh_clipping: float = 10.0, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, normalization: str = 'batch', norm_learn_affine: bool = True, norm_track_stats: bool = False, norm_eps_alpha: float = 1e-05, norm_momentum_beta: float = 0.1, lrnorm_k: float = 1.0, gnorm_groups: int = 3, activation_function: str = 'gelu', af_param: float = 1.0, af_threshold: float = 6.0, af_replacement_value: float = 6.0, af_num_params: int = 3, af_uniform_range: typing.List[float] = [0.125, 1 / 3], n_heads: int = 8, checkpoint_encoder: bool = False, shrink_size: typing.Optional[int] = None, pomo_size: int = 0, temporal_horizon: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, entropy_weight: float = 0.0, predictor_layers: typing.Optional[int] = None, connection_type: str = 'residual', hyper_expansion: int = 4)
:canonical: src.models.attention_model.AttentionModel

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.attention_model.AttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.AttentionModel.__init__
```

````{py:method} set_decode_type(decode_type: str, temp: typing.Optional[float] = None) -> None
:canonical: src.models.attention_model.AttentionModel.set_decode_type

```{autodoc2-docstring} src.models.attention_model.AttentionModel.set_decode_type
```

````

````{py:method} _get_initial_embeddings(input: typing.Dict[str, torch.Tensor]) -> torch.Tensor
:canonical: src.models.attention_model.AttentionModel._get_initial_embeddings

```{autodoc2-docstring} src.models.attention_model.AttentionModel._get_initial_embeddings
```

````

````{py:method} forward(input: typing.Dict[str, typing.Any], cost_weights: typing.Optional[torch.Tensor] = None, return_pi: bool = False, pad: bool = False, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Dict[str, torch.Tensor], typing.Optional[torch.Tensor], typing.Optional[torch.Tensor]]
:canonical: src.models.attention_model.AttentionModel.forward

```{autodoc2-docstring} src.models.attention_model.AttentionModel.forward
```

````

````{py:method} beam_search(*args: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: src.models.attention_model.AttentionModel.beam_search

```{autodoc2-docstring} src.models.attention_model.AttentionModel.beam_search
```

````

````{py:method} precompute_fixed(input: typing.Dict[str, torch.Tensor], edges: typing.Optional[torch.Tensor]) -> logic.src.utils.functions.beam_search.CachedLookup
:canonical: src.models.attention_model.AttentionModel.precompute_fixed

```{autodoc2-docstring} src.models.attention_model.AttentionModel.precompute_fixed
```

````

````{py:method} propose_expansions(beam: typing.Any, fixed: typing.Any, expand_size: typing.Optional[int] = None, normalize: bool = False, max_calc_batch_size: int = 4096) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.attention_model.AttentionModel.propose_expansions

```{autodoc2-docstring} src.models.attention_model.AttentionModel.propose_expansions
```

````

````{py:method} sample_many(input: typing.Dict[str, typing.Any], cost_weights: typing.Optional[torch.Tensor] = None, batch_rep: int = 1, iter_rep: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.attention_model.AttentionModel.sample_many

```{autodoc2-docstring} src.models.attention_model.AttentionModel.sample_many
```

````

`````
