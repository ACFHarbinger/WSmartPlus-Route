# {py:mod}`src.models.common.non_autoregressive.decoder`

```{py:module} src.models.common.non_autoregressive.decoder
```

```{autodoc2-docstring} src.models.common.non_autoregressive.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NonAutoregressiveDecoder <src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder>`
  - ```{autodoc2-docstring} src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder
    :summary:
    ```
````

### API

`````{py:class} NonAutoregressiveDecoder(**kwargs)
:canonical: src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, heatmap: torch.Tensor, env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder.forward
```

````

````{py:method} construct(td: tensordict.TensorDict, heatmap: torch.Tensor, env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Dict[str, torch.Tensor]
:canonical: src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder.construct

```{autodoc2-docstring} src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder.construct
```

````

`````
