# {py:mod}`src.models.common.non_autoregressive.encoder`

```{py:module} src.models.common.non_autoregressive.encoder
```

```{autodoc2-docstring} src.models.common.non_autoregressive.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NonAutoregressiveEncoder <src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder>`
  - ```{autodoc2-docstring} src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder
    :summary:
    ```
````

### API

`````{py:class} NonAutoregressiveEncoder(embed_dim: int = 128, **kwargs: typing.Any)
:canonical: src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs: typing.Any) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]
:canonical: src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder.forward
```

````

`````
