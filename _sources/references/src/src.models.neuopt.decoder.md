# {py:mod}`src.models.neuopt.decoder`

```{py:module} src.models.neuopt.decoder
```

```{autodoc2-docstring} src.models.neuopt.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOptDecoder <src.models.neuopt.decoder.NeuOptDecoder>`
  - ```{autodoc2-docstring} src.models.neuopt.decoder.NeuOptDecoder
    :summary:
    ```
````

### API

`````{py:class} NeuOptDecoder(embed_dim: int = 128, **kwargs)
:canonical: src.models.neuopt.decoder.NeuOptDecoder

Bases: {py:obj}`src.models.common.improvement_decoder.ImprovementDecoder`

```{autodoc2-docstring} src.models.neuopt.decoder.NeuOptDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.neuopt.decoder.NeuOptDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: torch.Tensor | typing.Tuple[torch.Tensor, ...], env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.neuopt.decoder.NeuOptDecoder.forward

```{autodoc2-docstring} src.models.neuopt.decoder.NeuOptDecoder.forward
```

````

`````
