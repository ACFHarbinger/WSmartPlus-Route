# {py:mod}`src.models.common.improvement.decoder`

```{py:module} src.models.common.improvement.decoder
```

```{autodoc2-docstring} src.models.common.improvement.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovementDecoder <src.models.common.improvement.decoder.ImprovementDecoder>`
  - ```{autodoc2-docstring} src.models.common.improvement.decoder.ImprovementDecoder
    :summary:
    ```
````

### API

`````{py:class} ImprovementDecoder(embed_dim: int = 128, **kwargs: typing.Any)
:canonical: src.models.common.improvement.decoder.ImprovementDecoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.improvement.decoder.ImprovementDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.improvement.decoder.ImprovementDecoder.__init__
```

````{py:property} device
:canonical: src.models.common.improvement.decoder.ImprovementDecoder.device
:type: torch.device

```{autodoc2-docstring} src.models.common.improvement.decoder.ImprovementDecoder.device
```

````

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.base.RL4COEnvBase, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.common.improvement.decoder.ImprovementDecoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.improvement.decoder.ImprovementDecoder.forward
```

````

`````
