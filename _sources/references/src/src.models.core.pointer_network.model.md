# {py:mod}`src.models.core.pointer_network.model`

```{py:module} src.models.core.pointer_network.model
```

```{autodoc2-docstring} src.models.core.pointer_network.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerNetwork <src.models.core.pointer_network.model.PointerNetwork>`
  - ```{autodoc2-docstring} src.models.core.pointer_network.model.PointerNetwork
    :summary:
    ```
````

### API

`````{py:class} PointerNetwork(embed_dim: int, hidden_dim: int, problem: typing.Any, n_encode_layers: typing.Optional[int] = None, tanh_clipping: float = 10.0, mask_inner: bool = True, mask_logits: bool = True, normalization: typing.Optional[str] = None, **kwargs: typing.Any)
:canonical: src.models.core.pointer_network.model.PointerNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.core.pointer_network.model.PointerNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.pointer_network.model.PointerNetwork.__init__
```

````{py:method} set_strategy(strategy: str) -> None
:canonical: src.models.core.pointer_network.model.PointerNetwork.set_strategy

```{autodoc2-docstring} src.models.core.pointer_network.model.PointerNetwork.set_strategy
```

````

````{py:method} forward(inputs: torch.Tensor, eval_tours: typing.Optional[torch.Tensor] = None, return_pi: bool = False) -> typing.Union[typing.Tuple[torch.Tensor, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
:canonical: src.models.core.pointer_network.model.PointerNetwork.forward

```{autodoc2-docstring} src.models.core.pointer_network.model.PointerNetwork.forward
```

````

````{py:method} _calc_log_likelihood(_log_p: torch.Tensor, a: torch.Tensor, mask: typing.Optional[torch.Tensor]) -> torch.Tensor
:canonical: src.models.core.pointer_network.model.PointerNetwork._calc_log_likelihood

```{autodoc2-docstring} src.models.core.pointer_network.model.PointerNetwork._calc_log_likelihood
```

````

````{py:method} _inner(inputs: torch.Tensor, eval_tours: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.pointer_network.model.PointerNetwork._inner

```{autodoc2-docstring} src.models.core.pointer_network.model.PointerNetwork._inner
```

````

`````
