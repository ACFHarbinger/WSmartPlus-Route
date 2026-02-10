# {py:mod}`src.models.subnets.decoders.matnet.decoder`

```{py:module} src.models.subnets.decoders.matnet.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.matnet.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MatNetDecoder <src.models.subnets.decoders.matnet.decoder.MatNetDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.matnet.decoder.MatNetDecoder
    :summary:
    ```
````

### API

`````{py:class} MatNetDecoder(embed_dim: int, hidden_dim: int, problem: typing.Any, n_heads: int = 8, tanh_clipping: float = 10.0, **kwargs)
:canonical: src.models.subnets.decoders.matnet.decoder.MatNetDecoder

Bases: {py:obj}`logic.src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder`

```{autodoc2-docstring} src.models.subnets.decoders.matnet.decoder.MatNetDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.matnet.decoder.MatNetDecoder.__init__
```

````{py:method} _precompute(embeddings: torch.Tensor, num_steps: int = 1) -> typing.Any
:canonical: src.models.subnets.decoders.matnet.decoder.MatNetDecoder._precompute

```{autodoc2-docstring} src.models.subnets.decoders.matnet.decoder.MatNetDecoder._precompute
```

````

````{py:method} forward(input: typing.Union[torch.Tensor, dict[str, torch.Tensor]], embeddings: torch.Tensor, cost_weights: typing.Optional[torch.Tensor] = None, dist_matrix: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.matnet.decoder.MatNetDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.matnet.decoder.MatNetDecoder.forward
```

````

`````
