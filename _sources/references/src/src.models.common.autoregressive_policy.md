# {py:mod}`src.models.common.autoregressive_policy`

```{py:module} src.models.common.autoregressive_policy
```

```{autodoc2-docstring} src.models.common.autoregressive_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoregressivePolicy <src.models.common.autoregressive_policy.AutoregressivePolicy>`
  - ```{autodoc2-docstring} src.models.common.autoregressive_policy.AutoregressivePolicy
    :summary:
    ```
````

### API

`````{py:class} AutoregressivePolicy(encoder: typing.Optional[src.models.common.autoregressive_encoder.AutoregressiveEncoder] = None, decoder: typing.Optional[src.models.common.autoregressive_decoder.AutoregressiveDecoder] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, **kwargs)
:canonical: src.models.common.autoregressive_policy.AutoregressivePolicy

Bases: {py:obj}`src.models.common.constructive.ConstructivePolicy`

```{autodoc2-docstring} src.models.common.autoregressive_policy.AutoregressivePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.autoregressive_policy.AutoregressivePolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.common.autoregressive_policy.AutoregressivePolicy.forward

```{autodoc2-docstring} src.models.common.autoregressive_policy.AutoregressivePolicy.forward
```

````

`````
