# {py:mod}`src.models.subnets.ptr_decoder`

```{py:module} src.models.subnets.ptr_decoder
```

```{autodoc2-docstring} src.models.subnets.ptr_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerAttention <src.models.subnets.ptr_decoder.PointerAttention>`
  - ```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerAttention
    :summary:
    ```
* - {py:obj}`PointerDecoder <src.models.subnets.ptr_decoder.PointerDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder
    :summary:
    ```
````

### API

`````{py:class} PointerAttention(dim, use_tanh=False, C=10)
:canonical: src.models.subnets.ptr_decoder.PointerAttention

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerAttention
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerAttention.__init__
```

````{py:method} forward(query, ref)
:canonical: src.models.subnets.ptr_decoder.PointerAttention.forward

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerAttention.forward
```

````

`````

`````{py:class} PointerDecoder(embedding_dim, hidden_dim, tanh_exploration, use_tanh, n_glimpses=1, mask_glimpses=True, mask_logits=True)
:canonical: src.models.subnets.ptr_decoder.PointerDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder.__init__
```

````{py:method} update_mask(mask, selected)
:canonical: src.models.subnets.ptr_decoder.PointerDecoder.update_mask

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder.update_mask
```

````

````{py:method} recurrence(x, h_in, prev_mask, prev_idxs, step, context)
:canonical: src.models.subnets.ptr_decoder.PointerDecoder.recurrence

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder.recurrence
```

````

````{py:method} calc_logits(x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None)
:canonical: src.models.subnets.ptr_decoder.PointerDecoder.calc_logits

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder.calc_logits
```

````

````{py:method} forward(decoder_input, embedded_inputs, hidden, context, eval_tours=None)
:canonical: src.models.subnets.ptr_decoder.PointerDecoder.forward

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder.forward
```

````

````{py:method} decode(probs, mask)
:canonical: src.models.subnets.ptr_decoder.PointerDecoder.decode

```{autodoc2-docstring} src.models.subnets.ptr_decoder.PointerDecoder.decode
```

````

`````
