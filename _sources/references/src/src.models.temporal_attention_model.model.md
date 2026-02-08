# {py:mod}`src.models.temporal_attention_model.model`

```{py:module} src.models.temporal_attention_model.model
```

```{autodoc2-docstring} src.models.temporal_attention_model.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalAttentionModel <src.models.temporal_attention_model.model.TemporalAttentionModel>`
  - ```{autodoc2-docstring} src.models.temporal_attention_model.model.TemporalAttentionModel
    :summary:
    ```
````

### API

`````{py:class} TemporalAttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, component_factory: logic.src.models.subnets.factories.NeuralComponentFactory, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, aggregation: str = 'sum', aggregation_graph: str = 'mean', tanh_clipping: float = 10.0, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, n_heads: int = 8, checkpoint_encoder: bool = False, shrink_size: typing.Optional[int] = None, temporal_horizon: int = 5, predictor_layers: int = 2, pomo_size: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, entropy_weight: float = 0.0, connection_type: str = 'residual', hyper_expansion: int = 4, decoder_type: str = 'attention', predictor_type: str = 'gru', **kwargs)
:canonical: src.models.temporal_attention_model.model.TemporalAttentionModel

Bases: {py:obj}`logic.src.models.attention_model.AttentionModel`

```{autodoc2-docstring} src.models.temporal_attention_model.model.TemporalAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.temporal_attention_model.model.TemporalAttentionModel.__init__
```

````{py:method} _get_initial_embeddings(input)
:canonical: src.models.temporal_attention_model.model.TemporalAttentionModel._get_initial_embeddings

```{autodoc2-docstring} src.models.temporal_attention_model.model.TemporalAttentionModel._get_initial_embeddings
```

````

````{py:method} forward(input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None, **kwargs)
:canonical: src.models.temporal_attention_model.model.TemporalAttentionModel.forward

```{autodoc2-docstring} src.models.temporal_attention_model.model.TemporalAttentionModel.forward
```

````

````{py:method} update_fill_history(fill_history, new_fills)
:canonical: src.models.temporal_attention_model.model.TemporalAttentionModel.update_fill_history

```{autodoc2-docstring} src.models.temporal_attention_model.model.TemporalAttentionModel.update_fill_history
```

````

````{py:method} compute_simulator_day(input, graph, run_tsp=False)
:canonical: src.models.temporal_attention_model.model.TemporalAttentionModel.compute_simulator_day

```{autodoc2-docstring} src.models.temporal_attention_model.model.TemporalAttentionModel.compute_simulator_day
```

````

`````
