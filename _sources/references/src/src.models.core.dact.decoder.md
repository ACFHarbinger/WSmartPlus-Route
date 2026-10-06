# {py:mod}`src.models.core.dact.decoder`

```{py:module} src.models.core.dact.decoder
```

```{autodoc2-docstring} src.models.core.dact.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DACTDecoder <src.models.core.dact.decoder.DACTDecoder>`
  - ```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder
    :summary:
    ```
````

### API

`````{py:class} DACTDecoder(embed_dim: int = 128, num_heads: int = 4, tanh_clipping: float = 6.0, seed: int = 42, **kwargs: typing.Any)
:canonical: src.models.core.dact.decoder.DACTDecoder

Bases: {py:obj}`logic.src.models.common.improvement.policy.ImprovementDecoder`

```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder.__init__
```

````{py:property} device
:canonical: src.models.core.dact.decoder.DACTDecoder.device
:type: torch.device

```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder.device
```

````

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.core.dact.decoder.DACTDecoder.__getstate__

```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.core.dact.decoder.DACTDecoder.__setstate__

```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder.__setstate__
```

````

````{py:method} _load_from_state_dict(state_dict: typing.Dict[str, typing.Any], prefix: str, local_metadata: typing.Dict[str, typing.Any], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]) -> None
:canonical: src.models.core.dact.decoder.DACTDecoder._load_from_state_dict

```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder._load_from_state_dict
```

````

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.base.RL4COEnvBase, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.dact.decoder.DACTDecoder.forward

```{autodoc2-docstring} src.models.core.dact.decoder.DACTDecoder.forward
```

````

`````
