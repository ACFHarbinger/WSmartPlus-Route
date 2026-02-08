# {py:mod}`src.utils.decoding.base`

```{py:module} src.utils.decoding.base
```

```{autodoc2-docstring} src.utils.decoding.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DecodingStrategy <src.utils.decoding.base.DecodingStrategy>`
  - ```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy
    :summary:
    ```
````

### API

`````{py:class} DecodingStrategy(temperature: float = 1.0, top_k: typing.Optional[int] = None, top_p: typing.Optional[float] = None, tanh_clipping: float = 0.0, mask_logits: bool = True, multistart: bool = False, num_starts: int = 1, select_best: bool = False, **kwargs)
:canonical: src.utils.decoding.base.DecodingStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.decoding.base.DecodingStrategy.step
:abstractmethod:

```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy.step
```

````

````{py:method} pre_decoder_hook(td: tensordict.TensorDict, env: typing.Any) -> typing.Tuple[tensordict.TensorDict, typing.Any, int]
:canonical: src.utils.decoding.base.DecodingStrategy.pre_decoder_hook

```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy.pre_decoder_hook
```

````

````{py:method} post_decoder_hook(td: tensordict.TensorDict, env: typing.Any, log_likelihood: torch.Tensor, actions: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, tensordict.TensorDict, typing.Any]
:canonical: src.utils.decoding.base.DecodingStrategy.post_decoder_hook

```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy.post_decoder_hook
```

````

````{py:method} _select_best(td: tensordict.TensorDict, log_likelihood: torch.Tensor, actions: torch.Tensor, num_starts: int) -> typing.Tuple[torch.Tensor, torch.Tensor, tensordict.TensorDict]
:canonical: src.utils.decoding.base.DecodingStrategy._select_best

```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy._select_best
```

````

````{py:method} _process_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
:canonical: src.utils.decoding.base.DecodingStrategy._process_logits

```{autodoc2-docstring} src.utils.decoding.base.DecodingStrategy._process_logits
```

````

`````
