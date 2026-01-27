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

`````{py:class} DeepDecoderAttentionModel(embedding_dim, hidden_dim, problem, encoder_class, n_encode_layers=2, n_encode_sublayers=None, n_decode_layers=2, dropout_rate=0.1, aggregation='sum', aggregation_graph='avg', tanh_clipping=10.0, mask_inner=True, mask_logits=True, mask_graph=False, normalization='batch', norm_learn_affine=True, norm_track_stats=False, norm_eps_alpha=1e-05, norm_momentum_beta=0.1, lrnorm_k=1.0, gnorm_groups=3, activation_function='gelu', af_param=1.0, af_threshold=6.0, af_replacement_value=6.0, af_num_params=3, af_uniform_range=[0.125, 1 / 3], n_heads=8, checkpoint_encoder=False, shrink_size=None, temporal_horizon=0, predictor_layers=None)
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
