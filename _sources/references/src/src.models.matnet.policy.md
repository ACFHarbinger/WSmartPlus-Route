# {py:mod}`src.models.matnet.policy`

```{py:module} src.models.matnet.policy
```

```{autodoc2-docstring} src.models.matnet.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MatNetPolicy <src.models.matnet.policy.MatNetPolicy>`
  - ```{autodoc2-docstring} src.models.matnet.policy.MatNetPolicy
    :summary:
    ```
````

### API

`````{py:class} MatNetPolicy(embed_dim: int, hidden_dim: int, problem: typing.Any, num_layers: int = 5, n_heads: int = 8, tanh_clipping: float = 10.0, normalization: str = 'instance', **kwargs)
:canonical: src.models.matnet.policy.MatNetPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.matnet.policy.MatNetPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.matnet.policy.MatNetPolicy.__init__
```

````{py:method} set_decode_type(decode_type: str, temp: typing.Optional[float] = None)
:canonical: src.models.matnet.policy.MatNetPolicy.set_decode_type

```{autodoc2-docstring} src.models.matnet.policy.MatNetPolicy.set_decode_type
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, decode_type: str = 'sampling', num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.matnet.policy.MatNetPolicy.forward

```{autodoc2-docstring} src.models.matnet.policy.MatNetPolicy.forward
```

````

`````
