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

`````{py:class} TemporalAttentionModel(embedding_dim, hidden_dim, problem, component_factory, n_encode_layers=2, n_encode_sublayers=None, n_decode_layers=None, dropout_rate=0.1, aggregation='sum', aggregation_graph='mean', tanh_clipping=10.0, mask_inner=True, mask_logits=True, mask_graph=False, normalization='batch', norm_learn_affine=True, norm_track_stats=False, norm_eps_alpha=1e-05, norm_momentum_beta=0.1, lrnorm_k=1.0, gnorm_groups=3, activation_function='gelu', af_param=1.0, af_threshold=6.0, af_replacement_value=6.0, af_num_params=3, af_uniform_range=[0.125, 1 / 3], n_heads=8, checkpoint_encoder=False, shrink_size=None, temporal_horizon=5, predictor_layers=2)
:canonical: src.models.temporal_am.TemporalAttentionModel

Bases: {py:obj}`src.models.AttentionModel`

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.temporal_am.TemporalAttentionModel.__init__
```

````{py:method} _get_initial_embeddings(nodes)
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
