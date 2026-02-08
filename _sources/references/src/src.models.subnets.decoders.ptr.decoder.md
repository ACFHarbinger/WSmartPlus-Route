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

`````{py:class} PointerDecoder(embed_dim, hidden_dim, tanh_exploration, use_tanh, n_glimpses=1, mask_glimpses=True, mask_logits=True)
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.__init__
```

````{py:method} update_mask(mask, selected)
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.update_mask

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.update_mask
```

````

````{py:method} recurrence(x, h_in, prev_mask, prev_idxs, step, context)
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.recurrence

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.recurrence
```

````

````{py:method} calc_logits(x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None)
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.calc_logits

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.calc_logits
```

````

````{py:method} forward(decoder_input, embedded_inputs, hidden, context, eval_tours=None)
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.forward
```

````

````{py:method} decode(probs, mask)
:canonical: src.models.subnets.decoders.ptr.decoder.PointerDecoder.decode

```{autodoc2-docstring} src.models.subnets.decoders.ptr.decoder.PointerDecoder.decode
```

````

`````
