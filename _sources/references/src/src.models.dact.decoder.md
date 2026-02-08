# {py:mod}`src.models.dact.decoder`

```{py:module} src.models.dact.decoder
```

```{autodoc2-docstring} src.models.dact.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DACTDecoder <src.models.dact.decoder.DACTDecoder>`
  - ```{autodoc2-docstring} src.models.dact.decoder.DACTDecoder
    :summary:
    ```
````

### API

`````{py:class} DACTDecoder(embed_dim: int = 128, num_heads: int = 8, **kwargs)
:canonical: src.models.dact.decoder.DACTDecoder

Bases: {py:obj}`src.models.common.improvement_policy.ImprovementDecoder`

```{autodoc2-docstring} src.models.dact.decoder.DACTDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.dact.decoder.DACTDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: torch.Tensor | typing.Tuple[torch.Tensor, ...], env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.dact.decoder.DACTDecoder.forward

```{autodoc2-docstring} src.models.dact.decoder.DACTDecoder.forward
```

````

`````
