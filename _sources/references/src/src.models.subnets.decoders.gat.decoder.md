# {py:mod}`src.models.subnets.decoders.gat.decoder`

```{py:module} src.models.subnets.decoders.gat.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepGATDecoder <src.models.subnets.decoders.gat.decoder.DeepGATDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder
    :summary:
    ```
````

### API

`````{py:class} DeepGATDecoder(embed_dim: int, hidden_dim: int, n_heads: int, n_layers: int, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, dropout_rate: float = 0.1, aggregation_graph: str = 'avg', mask_graph: bool = False, mask_logits: bool = True, tanh_clipping: float = 10.0, temp: float = 1.0, **kwargs)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder.__init__
```

````{py:method} forward(input: torch.Tensor, embeddings: torch.Tensor, fixed_context: typing.Optional[torch.Tensor] = None, init_context: typing.Optional[torch.Tensor] = None, env: typing.Optional[typing.Any] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder.forward
```

````

````{py:method} _inner(nodes: torch.Tensor, embeddings: torch.Tensor, fixed_context: typing.Optional[torch.Tensor] = None, init_context: typing.Optional[torch.Tensor] = None, env: typing.Optional[typing.Any] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._inner

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._inner
```

````

````{py:method} _select_node(probs, mask, strategy='greedy')
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._select_node

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._select_node
```

````

````{py:method} _precompute(embeddings, num_steps=1)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._precompute

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._precompute
```

````

````{py:method} _get_log_p(fixed, state, normalize=True)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_log_p

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_log_p
```

````

````{py:method} _one_to_many_logits(query, mha_K, mask, graph_mask)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._one_to_many_logits

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._one_to_many_logits
```

````

````{py:method} _get_parallel_step_context(embeddings, state)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_parallel_step_context

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_parallel_step_context
```

````

`````
