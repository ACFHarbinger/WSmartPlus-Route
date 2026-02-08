# {py:mod}`src.models.matnet.model`

```{py:module} src.models.matnet.model
```

```{autodoc2-docstring} src.models.matnet.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MatNet <src.models.matnet.model.MatNet>`
  - ```{autodoc2-docstring} src.models.matnet.model.MatNet
    :summary:
    ```
````

### API

`````{py:class} MatNet(embed_dim: int = 256, hidden_dim: int = 512, num_layers: int = 5, n_heads: int = 8, tanh_clipping: float = 10.0, normalization: str = 'instance', baseline: str = 'rollout', env_name: typing.Optional[str] = None, **kwargs)
:canonical: src.models.matnet.model.MatNet

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.matnet.model.MatNet
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.matnet.model.MatNet.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.matnet.model.MatNet.forward

```{autodoc2-docstring} src.models.matnet.model.MatNet.forward
```

````

````{py:method} set_decode_type(decode_type: str, **kwargs)
:canonical: src.models.matnet.model.MatNet.set_decode_type

```{autodoc2-docstring} src.models.matnet.model.MatNet.set_decode_type
```

````

`````
