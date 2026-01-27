# {py:mod}`src.models.pointer_network`

```{py:module} src.models.pointer_network
```

```{autodoc2-docstring} src.models.pointer_network
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerNetwork <src.models.pointer_network.PointerNetwork>`
  - ```{autodoc2-docstring} src.models.pointer_network.PointerNetwork
    :summary:
    ```
````

### API

`````{py:class} PointerNetwork(embedding_dim, hidden_dim, problem, n_encode_layers=None, tanh_clipping=10.0, mask_inner=True, mask_logits=True, normalization=None, **kwargs)
:canonical: src.models.pointer_network.PointerNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.pointer_network.PointerNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.pointer_network.PointerNetwork.__init__
```

````{py:method} set_decode_type(decode_type)
:canonical: src.models.pointer_network.PointerNetwork.set_decode_type

```{autodoc2-docstring} src.models.pointer_network.PointerNetwork.set_decode_type
```

````

````{py:method} forward(inputs, eval_tours=None, return_pi=False)
:canonical: src.models.pointer_network.PointerNetwork.forward

```{autodoc2-docstring} src.models.pointer_network.PointerNetwork.forward
```

````

````{py:method} _calc_log_likelihood(_log_p, a, mask)
:canonical: src.models.pointer_network.PointerNetwork._calc_log_likelihood

```{autodoc2-docstring} src.models.pointer_network.PointerNetwork._calc_log_likelihood
```

````

````{py:method} _inner(inputs, eval_tours=None)
:canonical: src.models.pointer_network.PointerNetwork._inner

```{autodoc2-docstring} src.models.pointer_network.PointerNetwork._inner
```

````

`````
