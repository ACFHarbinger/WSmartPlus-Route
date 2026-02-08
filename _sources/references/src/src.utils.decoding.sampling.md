# {py:mod}`src.utils.decoding.sampling`

```{py:module} src.utils.decoding.sampling
```

```{autodoc2-docstring} src.utils.decoding.sampling
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Sampling <src.utils.decoding.sampling.Sampling>`
  - ```{autodoc2-docstring} src.utils.decoding.sampling.Sampling
    :summary:
    ```
````

### API

`````{py:class} Sampling(temperature: float = 1.0, top_k: typing.Optional[int] = None, top_p: typing.Optional[float] = None, tanh_clipping: float = 0.0, mask_logits: bool = True, multistart: bool = False, num_starts: int = 1, select_best: bool = False, **kwargs)
:canonical: src.utils.decoding.sampling.Sampling

Bases: {py:obj}`src.utils.decoding.base.DecodingStrategy`

```{autodoc2-docstring} src.utils.decoding.sampling.Sampling
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.decoding.sampling.Sampling.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.decoding.sampling.Sampling.step

```{autodoc2-docstring} src.utils.decoding.sampling.Sampling.step
```

````

`````
