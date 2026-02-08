# {py:mod}`src.models.common.improvement_decoder`

```{py:module} src.models.common.improvement_decoder
```

```{autodoc2-docstring} src.models.common.improvement_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovementDecoder <src.models.common.improvement_decoder.ImprovementDecoder>`
  - ```{autodoc2-docstring} src.models.common.improvement_decoder.ImprovementDecoder
    :summary:
    ```
````

### API

`````{py:class} ImprovementDecoder(embed_dim: int = 128, **kwargs)
:canonical: src.models.common.improvement_decoder.ImprovementDecoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.improvement_decoder.ImprovementDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.improvement_decoder.ImprovementDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.common.improvement_decoder.ImprovementDecoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.improvement_decoder.ImprovementDecoder.forward
```

````

`````
