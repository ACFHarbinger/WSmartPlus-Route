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

````{py:property} device
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.device
:type: torch.device

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.device
```

````

````{py:method} __getstate__()
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.__getstate__

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.__getstate__
```

````

````{py:method} __setstate__(state)
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.__setstate__

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.__setstate__
```

````

````{py:method} forward(td: tensordict.TensorDict, embeddings: torch.Tensor | typing.Tuple[torch.Tensor, ...], env: logic.src.envs.base.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.forward

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.forward
```

````

`````
