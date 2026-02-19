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

`````{py:class} N2SDecoder(embed_dim: int = 128, **kwargs)
:canonical: src.models.core.n2s.decoder.N2SDecoder

Bases: {py:obj}`logic.src.models.common.improvement.decoder.ImprovementDecoder`

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: torch.Tensor | typing.Tuple[torch.Tensor, ...], env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.n2s.decoder.N2SDecoder.forward

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.forward
```

````

`````
