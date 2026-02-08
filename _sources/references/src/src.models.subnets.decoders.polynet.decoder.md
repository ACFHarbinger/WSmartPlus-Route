# {py:mod}`src.models.subnets.decoders.polynet.decoder`

```{py:module} src.models.subnets.decoders.polynet.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.polynet.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolyNetDecoder <src.models.subnets.decoders.polynet.decoder.PolyNetDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.polynet.decoder.PolyNetDecoder
    :summary:
    ```
````

### API

`````{py:class} PolyNetDecoder(k: int, encoder_type: str = 'AM', embed_dim: int = 128, poly_layer_dim: int = 256, num_heads: int = 8, env_name: str = 'vrpp', mask_inner: bool = True, out_bias: bool = False, linear_bias: bool = False, use_graph_context: bool = True, check_nan: bool = True)
:canonical: src.models.subnets.decoders.polynet.decoder.PolyNetDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.polynet.decoder.PolyNetDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.polynet.decoder.PolyNetDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: torch.Tensor, env: logic.src.envs.base.RL4COEnvBase, strategy: str = 'sampling', num_starts: int = 1, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.polynet.decoder.PolyNetDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.polynet.decoder.PolyNetDecoder.forward
```

````

````{py:method} _precompute_cache(embeddings: torch.Tensor) -> src.models.subnets.decoders.common.AttentionDecoderCache
:canonical: src.models.subnets.decoders.polynet.decoder.PolyNetDecoder._precompute_cache

```{autodoc2-docstring} src.models.subnets.decoders.polynet.decoder.PolyNetDecoder._precompute_cache
```

````

````{py:method} _get_step_logits(cache: src.models.subnets.decoders.common.AttentionDecoderCache, td: tensordict.TensorDict) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.polynet.decoder.PolyNetDecoder._get_step_logits

```{autodoc2-docstring} src.models.subnets.decoders.polynet.decoder.PolyNetDecoder._get_step_logits
```

````

`````
