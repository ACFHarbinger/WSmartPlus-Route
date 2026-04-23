# {py:mod}`src.models.core.mdam.policy`

```{py:module} src.models.core.mdam.policy
```

```{autodoc2-docstring} src.models.core.mdam.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MDAMPolicy <src.models.core.mdam.policy.MDAMPolicy>`
  - ```{autodoc2-docstring} src.models.core.mdam.policy.MDAMPolicy
    :summary:
    ```
````

### API

`````{py:class} MDAMPolicy(encoder: typing.Optional[logic.src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder] = None, decoder: typing.Optional[logic.src.models.subnets.decoders.mdam.MDAMDecoder] = None, embed_dim: int = 128, env_name: str = 'vrpp', num_encoder_layers: int = 3, num_heads: int = 8, num_paths: int = 5, normalization: str = 'batch', train_strategy: str = 'sampling', val_strategy: str = 'greedy', test_strategy: str = 'greedy', **decoder_kwargs: typing.Any)
:canonical: src.models.core.mdam.policy.MDAMPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive.policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.core.mdam.policy.MDAMPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.mdam.policy.MDAMPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.base.RL4COEnvBase, strategy: str = 'sampling', num_starts: int = 1, phase: str = 'train', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.mdam.policy.MDAMPolicy.forward

```{autodoc2-docstring} src.models.core.mdam.policy.MDAMPolicy.forward
```

````

`````
