# {py:mod}`src.models.core.n2s.decoder`

```{py:module} src.models.core.n2s.decoder
```

```{autodoc2-docstring} src.models.core.n2s.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`N2SDecoder <src.models.core.n2s.decoder.N2SDecoder>`
  - ```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder
    :summary:
    ```
````

### API

`````{py:class} N2SDecoder(embed_dim: int = 128, seed: int = 42, **kwargs: typing.Any)
:canonical: src.models.core.n2s.decoder.N2SDecoder

Bases: {py:obj}`logic.src.models.common.improvement.decoder.ImprovementDecoder`

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.__init__
```

````{py:property} device
:canonical: src.models.core.n2s.decoder.N2SDecoder.device
:type: torch.device

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.device
```

````

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.core.n2s.decoder.N2SDecoder.__getstate__

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.core.n2s.decoder.N2SDecoder.__setstate__

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.__setstate__
```

````

````{py:method} forward(td: tensordict.TensorDict, embeddings: torch.Tensor | typing.Tuple[torch.Tensor, ...], env: logic.src.envs.base.base.RL4COEnvBase, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.n2s.decoder.N2SDecoder.forward

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.forward
```

````

`````
