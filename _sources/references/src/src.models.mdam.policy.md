# {py:mod}`src.models.mdam.policy`

```{py:module} src.models.mdam.policy
```

```{autodoc2-docstring} src.models.mdam.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MDAMPolicy <src.models.mdam.policy.MDAMPolicy>`
  - ```{autodoc2-docstring} src.models.mdam.policy.MDAMPolicy
    :summary:
    ```
````

### API

`````{py:class} MDAMPolicy(encoder: typing.Optional[logic.src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder] = None, decoder: typing.Optional[logic.src.models.subnets.decoders.mdam.MDAMDecoder] = None, embed_dim: int = 128, env_name: str = 'vrpp', num_encoder_layers: int = 3, num_heads: int = 8, num_paths: int = 5, normalization: str = 'batch', train_decode_type: str = 'sampling', val_decode_type: str = 'greedy', test_decode_type: str = 'greedy', **decoder_kwargs)
:canonical: src.models.mdam.policy.MDAMPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.mdam.policy.MDAMPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.mdam.policy.MDAMPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, phase: str = 'train', **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.mdam.policy.MDAMPolicy.forward

```{autodoc2-docstring} src.models.mdam.policy.MDAMPolicy.forward
```

````

`````
