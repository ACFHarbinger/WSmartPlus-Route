# {py:mod}`src.models.common.autoregressive.decoder`

```{py:module} src.models.common.autoregressive.decoder
```

```{autodoc2-docstring} src.models.common.autoregressive.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoregressiveDecoder <src.models.common.autoregressive.decoder.AutoregressiveDecoder>`
  - ```{autodoc2-docstring} src.models.common.autoregressive.decoder.AutoregressiveDecoder
    :summary:
    ```
````

### API

`````{py:class} AutoregressiveDecoder(embed_dim: int = 128, **kwargs: typing.Any)
:canonical: src.models.common.autoregressive.decoder.AutoregressiveDecoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.autoregressive.decoder.AutoregressiveDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.autoregressive.decoder.AutoregressiveDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.common.autoregressive.decoder.AutoregressiveDecoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.autoregressive.decoder.AutoregressiveDecoder.forward
```

````

`````
