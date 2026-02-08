# {py:mod}`src.models.subnets.decoders.glimpse.decoder`

```{py:module} src.models.subnets.decoders.glimpse.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GlimpseDecoder <src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder
    :summary:
    ```
````

### API

`````{py:class} GlimpseDecoder(embed_dim: int, hidden_dim: int, problem: typing.Any, n_heads: int = 8, mask_inner: bool = True, mask_logits: bool = True, tanh_clipping: float = 10.0, mask_graph: bool = False, shrink_size: typing.Optional[int] = None, pomo_size: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, decode_type: typing.Optional[str] = None, **kwargs)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.__init__
```

````{py:method} set_step_context_dim(dim: int)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.set_step_context_dim

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.set_step_context_dim
```

````

````{py:method} set_decode_type(decode_type: str, temp: typing.Optional[float] = None)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.set_decode_type

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.set_decode_type
```

````

````{py:method} forward(input: typing.Union[torch.Tensor, dict[str, torch.Tensor]], embeddings: torch.Tensor, cost_weights: typing.Optional[torch.Tensor] = None, dist_matrix: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.forward
```

````

````{py:method} _inner(nodes: typing.Union[torch.Tensor, dict[str, torch.Tensor]], edges: typing.Optional[torch.Tensor], embeddings: torch.Tensor, cost_weights: typing.Optional[torch.Tensor], dist_matrix: typing.Optional[torch.Tensor], profit_vars: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._inner

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._inner
```

````

````{py:method} _select_node(probs: torch.Tensor, mask: typing.Optional[torch.Tensor], decode_type: str = 'greedy')
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._select_node

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._select_node
```

````

````{py:method} _precompute(embeddings: torch.Tensor, num_steps: int = 1) -> src.models.subnets.decoders.glimpse.fixed.AttentionModelFixed
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._precompute

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._precompute
```

````

````{py:method} _get_log_p(fixed: src.models.subnets.decoders.glimpse.fixed.AttentionModelFixed, state: typing.Any, normalize: bool = True, mask_val: float = -math.inf, mask: typing.Optional[torch.Tensor] = None)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._get_log_p

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._get_log_p
```

````

````{py:method} _get_parallel_step_context(embeddings: torch.Tensor, state: typing.Any, from_depot: bool = False)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._get_parallel_step_context

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._get_parallel_step_context
```

````

````{py:method} _calc_log_likelihood(_log_p: torch.Tensor, a: torch.Tensor, mask: typing.Optional[torch.Tensor], return_entropy: bool = False, kl_loss: bool = False)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._calc_log_likelihood

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder._calc_log_likelihood
```

````

````{py:method} propose_expansions(beam: typing.Any, fixed: src.models.subnets.decoders.glimpse.fixed.AttentionModelFixed, expand_size: typing.Optional[int] = None, normalize: bool = False, max_calc_batch_size: int = 4096)
:canonical: src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.propose_expansions

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder.propose_expansions
```

````

`````
