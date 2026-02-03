# {py:mod}`src.models.temporal_am`

```{py:module} src.models.temporal_am
```

```{autodoc2-docstring} src.models.temporal_am
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalAttentionModel <src.models.temporal_am.TemporalAttentionModel>`
  - ```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel
    :summary:
    ```
````

### API

`````{py:class} TemporalAttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, component_factory: logic.src.models.model_factory.NeuralComponentFactory, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, aggregation: str = 'sum', aggregation_graph: str = 'mean', tanh_clipping: float = 10.0, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, normalization: str = 'batch', norm_learn_affine: bool = True, norm_track_stats: bool = False, norm_eps_alpha: float = 1e-05, norm_momentum_beta: float = 0.1, lrnorm_k: float = 1.0, gnorm_groups: int = 3, activation_function: str = 'gelu', af_param: float = 1.0, af_threshold: float = 6.0, af_replacement_value: float = 6.0, af_num_params: int = 3, af_uniform_range: typing.List[float] = [0.125, 1 / 3], n_heads: int = 8, checkpoint_encoder: bool = False, shrink_size: typing.Optional[int] = None, temporal_horizon: int = 5, predictor_layers: int = 2)
:canonical: src.models.temporal_am.TemporalAttentionModel

Bases: {py:obj}`src.models.AttentionModel`

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel.__init__
```

````{py:method} _get_initial_embeddings(input)
:canonical: src.models.temporal_am.TemporalAttentionModel._get_initial_embeddings

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel._get_initial_embeddings
```

````

````{py:method} forward(input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None, **kwargs)
:canonical: src.models.temporal_am.TemporalAttentionModel.forward

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel.forward
```

````

````{py:method} update_fill_history(fill_history, new_fills)
:canonical: src.models.temporal_am.TemporalAttentionModel.update_fill_history

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel.update_fill_history
```

````

````{py:method} compute_simulator_day(input, graph, run_tsp=False)
:canonical: src.models.temporal_am.TemporalAttentionModel.compute_simulator_day

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel.compute_simulator_day
```

````

`````
