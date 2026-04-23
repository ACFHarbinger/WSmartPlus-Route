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

`````{py:class} DeepGATDecoder(embed_dim: int, hidden_dim: int, n_heads: int, n_layers: int, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, dropout_rate: float = 0.1, aggregation_graph: str = 'avg', mask_graph: bool = False, mask_logits: bool = True, tanh_clipping: float = 10.0, seed: int = 42, temp: float = 1.0, **kwargs: typing.Any)
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder.__init__
```

````{py:property} device
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder.device
:type: torch.device

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder.device
```

````

````{py:method} forward(input: torch.Tensor | typing.Dict[str, torch.Tensor], embeddings: torch.Tensor, fixed_context: typing.Optional[torch.Tensor] = None, init_context: typing.Optional[torch.Tensor] = None, env: typing.Optional[typing.Any] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor], None]
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder.forward
```

````

````{py:method} _inner(nodes: torch.Tensor | typing.Dict[str, torch.Tensor], embeddings: torch.Tensor, fixed_context: typing.Optional[torch.Tensor] = None, init_context: typing.Optional[torch.Tensor] = None, env: typing.Optional[typing.Any] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor], None]
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._inner

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._inner
```

````

````{py:method} _select_node(probs: torch.Tensor, mask: torch.Tensor, strategy: str = 'greedy') -> torch.Tensor
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._select_node

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._select_node
```

````

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder.__getstate__

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder.__setstate__

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder.__setstate__
```

````

````{py:method} _precompute(embeddings: torch.Tensor, num_steps: int = 1) -> logic.src.models.subnets.decoders.common.AttentionDecoderCache
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._precompute

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._precompute
```

````

````{py:method} _get_log_p(fixed: logic.src.models.subnets.decoders.common.AttentionDecoderCache, state: typing.Any, normalize: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_log_p

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_log_p
```

````

````{py:method} _one_to_many_logits(query: torch.Tensor, mha_K: torch.Tensor, mask: torch.Tensor, graph_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._one_to_many_logits

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._one_to_many_logits
```

````

````{py:method} _get_parallel_step_context(embeddings: torch.Tensor, state: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_parallel_step_context

```{autodoc2-docstring} src.models.subnets.decoders.gat.decoder.DeepGATDecoder._get_parallel_step_context
```

````

`````
