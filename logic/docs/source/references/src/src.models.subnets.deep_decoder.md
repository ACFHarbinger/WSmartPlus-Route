# {py:mod}`src.models.subnets.deep_decoder`

```{py:module} src.models.subnets.deep_decoder
```

```{autodoc2-docstring} src.models.subnets.deep_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepAttentionModelFixed <src.models.subnets.deep_decoder.DeepAttentionModelFixed>`
  - ```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepAttentionModelFixed
    :summary:
    ```
* - {py:obj}`DeepDecoder <src.models.subnets.deep_decoder.DeepDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepDecoder
    :summary:
    ```
````

### API

`````{py:class} DeepAttentionModelFixed
:canonical: src.models.subnets.deep_decoder.DeepAttentionModelFixed

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepAttentionModelFixed
```

````{py:attribute} node_embeddings
:canonical: src.models.subnets.deep_decoder.DeepAttentionModelFixed.node_embeddings
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepAttentionModelFixed.node_embeddings
```

````

````{py:attribute} context_node_projected
:canonical: src.models.subnets.deep_decoder.DeepAttentionModelFixed.context_node_projected
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepAttentionModelFixed.context_node_projected
```

````

````{py:method} __getitem__(key)
:canonical: src.models.subnets.deep_decoder.DeepAttentionModelFixed.__getitem__

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepAttentionModelFixed.__getitem__
```

````

`````

`````{py:class} DeepDecoder(embed_dim: int, hidden_dim: int, n_heads: int, n_layers: int, normalization: str = 'batch', dropout_rate: float = 0.1, aggregation_graph: str = 'avg', mask_graph: bool = False, mask_logits: bool = True, tanh_clipping: float = 10.0, temp: float = 1.0, **kwargs)
:canonical: src.models.subnets.deep_decoder.DeepDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepDecoder.__init__
```

````{py:method} _precompute(embeddings, num_steps=1)
:canonical: src.models.subnets.deep_decoder.DeepDecoder._precompute

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepDecoder._precompute
```

````

````{py:method} _get_log_p(fixed, state, normalize=True)
:canonical: src.models.subnets.deep_decoder.DeepDecoder._get_log_p

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepDecoder._get_log_p
```

````

````{py:method} _one_to_many_logits(query, mha_K, mask, graph_mask)
:canonical: src.models.subnets.deep_decoder.DeepDecoder._one_to_many_logits

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepDecoder._one_to_many_logits
```

````

````{py:method} _get_parallel_step_context(embeddings, state)
:canonical: src.models.subnets.deep_decoder.DeepDecoder._get_parallel_step_context

```{autodoc2-docstring} src.models.subnets.deep_decoder.DeepDecoder._get_parallel_step_context
```

````

`````
