# {py:mod}`src.models.core.neuopt.decoder`

```{py:module} src.models.core.neuopt.decoder
```

```{autodoc2-docstring} src.models.core.neuopt.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOptDecoder <src.models.core.neuopt.decoder.NeuOptDecoder>`
  - ```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder
    :summary:
    ```
````

### API

`````{py:class} NeuOptDecoder(embed_dim: int = 128, seed: int = 42, **kwargs)
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder

Bases: {py:obj}`logic.src.models.common.improvement.decoder.ImprovementDecoder`

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: torch.Tensor | typing.Tuple[torch.Tensor, ...], env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.forward

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.forward
```

````

`````
