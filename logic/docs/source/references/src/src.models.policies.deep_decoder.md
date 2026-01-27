# {py:mod}`src.models.policies.deep_decoder`

```{py:module} src.models.policies.deep_decoder
```

```{autodoc2-docstring} src.models.policies.deep_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepDecoderPolicy <src.models.policies.deep_decoder.DeepDecoderPolicy>`
  - ```{autodoc2-docstring} src.models.policies.deep_decoder.DeepDecoderPolicy
    :summary:
    ```
````

### API

`````{py:class} DeepDecoderPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_encode_layers: int = 3, n_decode_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', dropout_rate: float = 0.1, **kwargs)
:canonical: src.models.policies.deep_decoder.DeepDecoderPolicy

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.deep_decoder.DeepDecoderPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.deep_decoder.DeepDecoderPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> dict
:canonical: src.models.policies.deep_decoder.DeepDecoderPolicy.forward

```{autodoc2-docstring} src.models.policies.deep_decoder.DeepDecoderPolicy.forward
```

````

`````
