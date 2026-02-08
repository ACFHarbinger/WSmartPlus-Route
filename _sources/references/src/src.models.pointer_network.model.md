# {py:mod}`src.models.pointer_network.model`

```{py:module} src.models.pointer_network.model
```

```{autodoc2-docstring} src.models.pointer_network.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerNetwork <src.models.pointer_network.model.PointerNetwork>`
  - ```{autodoc2-docstring} src.models.pointer_network.model.PointerNetwork
    :summary:
    ```
````

### API

`````{py:class} PointerNetwork(embed_dim, hidden_dim, problem, n_encode_layers=None, tanh_clipping=10.0, mask_inner=True, mask_logits=True, normalization=None, **kwargs)
:canonical: src.models.pointer_network.model.PointerNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.pointer_network.model.PointerNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.pointer_network.model.PointerNetwork.__init__
```

````{py:method} set_strategy(strategy)
:canonical: src.models.pointer_network.model.PointerNetwork.set_strategy

```{autodoc2-docstring} src.models.pointer_network.model.PointerNetwork.set_strategy
```

````

````{py:method} forward(inputs, eval_tours=None, return_pi=False)
:canonical: src.models.pointer_network.model.PointerNetwork.forward

```{autodoc2-docstring} src.models.pointer_network.model.PointerNetwork.forward
```

````

````{py:method} _calc_log_likelihood(_log_p, a, mask)
:canonical: src.models.pointer_network.model.PointerNetwork._calc_log_likelihood

```{autodoc2-docstring} src.models.pointer_network.model.PointerNetwork._calc_log_likelihood
```

````

````{py:method} _inner(inputs, eval_tours=None)
:canonical: src.models.pointer_network.model.PointerNetwork._inner

```{autodoc2-docstring} src.models.pointer_network.model.PointerNetwork._inner
```

````

`````
