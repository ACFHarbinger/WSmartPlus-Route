# {py:mod}`src.models.common.autoregressive.policy`

```{py:module} src.models.common.autoregressive.policy
```

```{autodoc2-docstring} src.models.common.autoregressive.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoregressivePolicy <src.models.common.autoregressive.policy.AutoregressivePolicy>`
  - ```{autodoc2-docstring} src.models.common.autoregressive.policy.AutoregressivePolicy
    :summary:
    ```
````

### API

`````{py:class} AutoregressivePolicy(encoder: typing.Optional[src.models.common.autoregressive.encoder.AutoregressiveEncoder] = None, decoder: typing.Optional[src.models.common.autoregressive.decoder.AutoregressiveDecoder] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, seed: int = 42, device: str = 'cpu', **kwargs: typing.Any)
:canonical: src.models.common.autoregressive.policy.AutoregressivePolicy

Bases: {py:obj}`src.models.common.autoregressive.constructive.ConstructivePolicy`

```{autodoc2-docstring} src.models.common.autoregressive.policy.AutoregressivePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.autoregressive.policy.AutoregressivePolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, strategy: str = 'sampling', num_starts: int = 1, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.common.autoregressive.policy.AutoregressivePolicy.forward

```{autodoc2-docstring} src.models.common.autoregressive.policy.AutoregressivePolicy.forward
```

````

`````
