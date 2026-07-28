# {py:mod}`src.models.subnets.decoders.ptr.decoder`

```{py:module} src.models.subnets.decoders.ptr.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerDecoder <src.models.subnets.decoders.ptr.decoder.PointerDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder
    :summary:
    ```
````

### API

`````{py:class} PointerDecoder(embed_dim: int, hidden_dim: int, tanh_exploration: float, use_tanh: bool, n_glimpses: int = 1, mask_glimpses: bool = True, mask_logits: bool = True)
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.__init__
```

````{py:method} update_mask(mask: torch.Tensor, selected: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.update_mask

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.update_mask
```

````

````{py:method} recurrence(x: torch.Tensor, h_in: typing.Tuple[torch.Tensor, torch.Tensor], prev_mask: torch.Tensor, prev_idxs: typing.Optional[torch.Tensor], step: int, context: torch.Tensor) -> typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.recurrence

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.recurrence
```

````

````{py:method} calc_logits(x: torch.Tensor, h_in: typing.Tuple[torch.Tensor, torch.Tensor], logit_mask: torch.Tensor, context: torch.Tensor, mask_glimpses: typing.Optional[bool] = None, mask_logits: typing.Optional[bool] = None) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.calc_logits

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.calc_logits
```

````

````{py:method} forward(decoder_input: torch.Tensor, embedded_inputs: torch.Tensor, hidden: typing.Tuple[torch.Tensor, torch.Tensor], context: torch.Tensor, eval_tours: typing.Optional[torch.Tensor] = None) -> typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.forward
```

````

````{py:method} decode(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.decode

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.decode
```

````

`````
