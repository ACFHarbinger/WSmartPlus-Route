# {py:mod}`src.models.deep_decoder_am`

```{py:module} src.models.deep_decoder_am
```

```{autodoc2-docstring} src.models.deep_decoder_am
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepAttentionModelFixed <src.models.deep_decoder_am.DeepAttentionModelFixed>`
  - ```{autodoc2-docstring} src.models.deep_decoder_am.DeepAttentionModelFixed
    :summary:
    ```
* - {py:obj}`DeepDecoderAttentionModel <src.models.deep_decoder_am.DeepDecoderAttentionModel>`
  - ```{autodoc2-docstring} src.models.deep_decoder_am.DeepDecoderAttentionModel
    :summary:
    ```
````

### API

`````{py:class} DeepAttentionModelFixed
:canonical: src.models.deep_decoder_am.DeepAttentionModelFixed

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} src.models.deep_decoder_am.DeepAttentionModelFixed
```

````{py:attribute} node_embeddings
:canonical: src.models.deep_decoder_am.DeepAttentionModelFixed.node_embeddings
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.deep_decoder_am.DeepAttentionModelFixed.node_embeddings
```

````

````{py:attribute} context_node_projected
:canonical: src.models.deep_decoder_am.DeepAttentionModelFixed.context_node_projected
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.deep_decoder_am.DeepAttentionModelFixed.context_node_projected
```

````

````{py:attribute} mha_key
:canonical: src.models.deep_decoder_am.DeepAttentionModelFixed.mha_key
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.deep_decoder_am.DeepAttentionModelFixed.mha_key
```

````

````{py:method} __getitem__(key)
:canonical: src.models.deep_decoder_am.DeepAttentionModelFixed.__getitem__

```{autodoc2-docstring} src.models.deep_decoder_am.DeepAttentionModelFixed.__getitem__
```

````

`````

`````{py:class} DeepDecoderAttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, component_factory: logic.src.models.model_factory.NeuralComponentFactory, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: int = 2, dropout_rate: float = 0.1, aggregation: str = 'sum', aggregation_graph: str = 'avg', tanh_clipping: float = 10.0, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, normalization: str = 'batch', norm_learn_affine: bool = True, norm_track_stats: bool = False, norm_eps_alpha: float = 1e-05, norm_momentum_beta: float = 0.1, lrnorm_k: float = 1.0, gnorm_groups: int = 3, activation_function: str = 'gelu', af_param: float = 1.0, af_threshold: float = 6.0, af_replacement_value: float = 6.0, af_num_params: int = 3, af_uniform_range: typing.List[float] = [0.125, 1 / 3], n_heads: int = 8, checkpoint_encoder: bool = False, shrink_size: typing.Optional[int] = None, pomo_size: int = 0, temporal_horizon: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, entropy_weight: float = 0.0, predictor_layers: typing.Optional[int] = None, connection_type: str = 'residual', hyper_expansion: int = 4)
:canonical: src.models.deep_decoder_am.DeepDecoderAttentionModel

Bases: {py:obj}`src.models.AttentionModel`

```{autodoc2-docstring} src.models.deep_decoder_am.DeepDecoderAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.deep_decoder_am.DeepDecoderAttentionModel.__init__
```

````{py:method} _precompute(embeddings, num_steps=1)
:canonical: src.models.deep_decoder_am.DeepDecoderAttentionModel._precompute

```{autodoc2-docstring} src.models.deep_decoder_am.DeepDecoderAttentionModel._precompute
```

````

````{py:method} _get_log_p(fixed, state, normalize=True)
:canonical: src.models.deep_decoder_am.DeepDecoderAttentionModel._get_log_p

```{autodoc2-docstring} src.models.deep_decoder_am.DeepDecoderAttentionModel._get_log_p
```

````

````{py:method} _one_to_many_logits(query, mha_K, mask, graph_mask)
:canonical: src.models.deep_decoder_am.DeepDecoderAttentionModel._one_to_many_logits

```{autodoc2-docstring} src.models.deep_decoder_am.DeepDecoderAttentionModel._one_to_many_logits
```

````

````{py:method} _get_attention_node_data(fixed, state)
:canonical: src.models.deep_decoder_am.DeepDecoderAttentionModel._get_attention_node_data

```{autodoc2-docstring} src.models.deep_decoder_am.DeepDecoderAttentionModel._get_attention_node_data
```

````

`````
