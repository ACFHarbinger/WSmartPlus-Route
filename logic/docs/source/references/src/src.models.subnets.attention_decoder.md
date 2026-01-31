# {py:mod}`src.models.subnets.attention_decoder`

```{py:module} src.models.subnets.attention_decoder
```

```{autodoc2-docstring} src.models.subnets.attention_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionModelFixed <src.models.subnets.attention_decoder.AttentionModelFixed>`
  - ```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed
    :summary:
    ```
* - {py:obj}`AttentionDecoder <src.models.subnets.attention_decoder.AttentionDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder
    :summary:
    ```
````

### API

`````{py:class} AttentionModelFixed
:canonical: src.models.subnets.attention_decoder.AttentionModelFixed

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed
```

````{py:attribute} node_embeddings
:canonical: src.models.subnets.attention_decoder.AttentionModelFixed.node_embeddings
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed.node_embeddings
```

````

````{py:attribute} context_node_projected
:canonical: src.models.subnets.attention_decoder.AttentionModelFixed.context_node_projected
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed.context_node_projected
```

````

````{py:attribute} glimpse_key
:canonical: src.models.subnets.attention_decoder.AttentionModelFixed.glimpse_key
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed.glimpse_key
```

````

````{py:attribute} glimpse_val
:canonical: src.models.subnets.attention_decoder.AttentionModelFixed.glimpse_val
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed.glimpse_val
```

````

````{py:attribute} logit_key
:canonical: src.models.subnets.attention_decoder.AttentionModelFixed.logit_key
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed.logit_key
```

````

````{py:method} __getitem__(key: typing.Union[int, slice, torch.Tensor]) -> src.models.subnets.attention_decoder.AttentionModelFixed
:canonical: src.models.subnets.attention_decoder.AttentionModelFixed.__getitem__

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionModelFixed.__getitem__
```

````

`````

`````{py:class} AttentionDecoder(embed_dim: int, hidden_dim: int, problem: typing.Any, n_heads: int = 8, tanh_clipping: float = 10.0, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, shrink_size: typing.Optional[int] = None, pomo_size: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, decode_type: typing.Optional[str] = None, **kwargs)
:canonical: src.models.subnets.attention_decoder.AttentionDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder.__init__
```

````{py:method} set_step_context_dim(dim: int) -> None
:canonical: src.models.subnets.attention_decoder.AttentionDecoder.set_step_context_dim

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder.set_step_context_dim
```

````

````{py:method} set_decode_type(decode_type: str, temp: typing.Optional[float] = None) -> None
:canonical: src.models.subnets.attention_decoder.AttentionDecoder.set_decode_type

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder.set_decode_type
```

````

````{py:method} forward(input: typing.Union[torch.Tensor, dict[str, torch.Tensor]], embeddings: torch.Tensor, cost_weights: typing.Optional[torch.Tensor] = None, dist_matrix: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder.forward

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder.forward
```

````

````{py:method} _inner(nodes: typing.Union[torch.Tensor, dict[str, torch.Tensor]], edges: typing.Optional[torch.Tensor], embeddings: torch.Tensor, cost_weights: typing.Optional[torch.Tensor], dist_matrix: typing.Optional[torch.Tensor], profit_vars: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._inner

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._inner
```

````

````{py:method} _select_node(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._select_node

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._select_node
```

````

````{py:method} _precompute(embeddings: torch.Tensor, num_steps: int = 1) -> src.models.subnets.attention_decoder.AttentionModelFixed
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._precompute

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._precompute
```

````

````{py:method} _make_heads(v: torch.Tensor, num_steps: typing.Optional[int] = None) -> torch.Tensor
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._make_heads

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._make_heads
```

````

````{py:method} _get_log_p(fixed: src.models.subnets.attention_decoder.AttentionModelFixed, state: typing.Any, normalize: bool = True, mask_val: float = -math.inf) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._get_log_p

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._get_log_p
```

````

````{py:method} _get_parallel_step_context(embeddings: torch.Tensor, state: typing.Any, from_depot: bool = False) -> torch.Tensor
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._get_parallel_step_context

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._get_parallel_step_context
```

````

````{py:method} _get_attention_node_data(fixed: src.models.subnets.attention_decoder.AttentionModelFixed, state: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._get_attention_node_data

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._get_attention_node_data
```

````

````{py:method} _one_to_many_logits(query: torch.Tensor, glimpse_K: torch.Tensor, glimpse_V: torch.Tensor, logit_K: torch.Tensor, mask: torch.Tensor, graph_mask: typing.Optional[torch.Tensor] = None, dist_bias: typing.Optional[torch.Tensor] = None, mask_val: float = -math.inf) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._one_to_many_logits

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._one_to_many_logits
```

````

````{py:method} _calc_log_likelihood(_log_p: torch.Tensor, a: torch.Tensor, mask: typing.Optional[torch.Tensor], return_entropy: bool = False, kl_loss: bool = False) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._calc_log_likelihood

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._calc_log_likelihood
```

````

````{py:method} _get_log_p_topk(fixed: src.models.subnets.attention_decoder.AttentionModelFixed, state: typing.Any, k: typing.Optional[int] = None, normalize: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder._get_log_p_topk

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder._get_log_p_topk
```

````

````{py:method} propose_expansions(beam: typing.Any, fixed: src.models.subnets.attention_decoder.AttentionModelFixed, expand_size: typing.Optional[int] = None, normalize: bool = False, max_calc_batch_size: int = 4096) -> typing.Tuple[typing.Optional[torch.Tensor], typing.Optional[torch.Tensor], typing.Optional[torch.Tensor]]
:canonical: src.models.subnets.attention_decoder.AttentionDecoder.propose_expansions

```{autodoc2-docstring} src.models.subnets.attention_decoder.AttentionDecoder.propose_expansions
```

````

`````
