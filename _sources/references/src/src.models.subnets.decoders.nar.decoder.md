# {py:mod}`src.models.subnets.decoders.nar.decoder`

```{py:module} src.models.subnets.decoders.nar.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.nar.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimpleNARDecoder <src.models.subnets.decoders.nar.decoder.SimpleNARDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.nar.decoder.SimpleNARDecoder
    :summary:
    ```
````

### API

`````{py:class} SimpleNARDecoder(**kwargs)
:canonical: src.models.subnets.decoders.nar.decoder.SimpleNARDecoder

Bases: {py:obj}`logic.src.models.common.nonautoregressive_decoder.NonAutoregressiveDecoder`

```{autodoc2-docstring} src.models.subnets.decoders.nar.decoder.SimpleNARDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.nar.decoder.SimpleNARDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, heatmap: torch.Tensor, env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.nar.decoder.SimpleNARDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.nar.decoder.SimpleNARDecoder.forward
```

````

`````
