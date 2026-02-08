# {py:mod}`src.models.common.autoregressive_decoder`

```{py:module} src.models.common.autoregressive_decoder
```

```{autodoc2-docstring} src.models.common.autoregressive_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoregressiveDecoder <src.models.common.autoregressive_decoder.AutoregressiveDecoder>`
  - ```{autodoc2-docstring} src.models.common.autoregressive_decoder.AutoregressiveDecoder
    :summary:
    ```
````

### API

`````{py:class} AutoregressiveDecoder(embed_dim: int = 128, **kwargs)
:canonical: src.models.common.autoregressive_decoder.AutoregressiveDecoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.autoregressive_decoder.AutoregressiveDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.autoregressive_decoder.AutoregressiveDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.common.autoregressive_decoder.AutoregressiveDecoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.autoregressive_decoder.AutoregressiveDecoder.forward
```

````

`````
