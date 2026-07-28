# {py:mod}`src.models.common.non_autoregressive.policy`

```{py:module} src.models.common.non_autoregressive.policy
```

```{autodoc2-docstring} src.models.common.non_autoregressive.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NonAutoregressivePolicy <src.models.common.non_autoregressive.policy.NonAutoregressivePolicy>`
  - ```{autodoc2-docstring} src.models.common.non_autoregressive.policy.NonAutoregressivePolicy
    :summary:
    ```
````

### API

`````{py:class} NonAutoregressivePolicy(encoder: typing.Optional[src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder] = None, decoder: typing.Optional[src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, **kwargs: typing.Any)
:canonical: src.models.common.non_autoregressive.policy.NonAutoregressivePolicy

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.non_autoregressive.policy.NonAutoregressivePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, strategy: str = 'sampling', num_starts: int = 1, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.forward

```{autodoc2-docstring} src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.forward
```

````

````{py:method} set_strategy(strategy: str, **kwargs: typing.Any) -> None
:canonical: src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.set_strategy

```{autodoc2-docstring} src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.set_strategy
```

````

````{py:method} common_decoding(strategy: str, td: tensordict.TensorDict, env: logic.src.envs.base.base.RL4COEnvBase, heatmap: torch.Tensor, actions: typing.Optional[torch.Tensor] = None, **decoding_kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor, tensordict.TensorDict, logic.src.envs.base.base.RL4COEnvBase]
:canonical: src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.common_decoding

```{autodoc2-docstring} src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.common_decoding
```

````

````{py:method} eval() -> src.models.common.non_autoregressive.policy.NonAutoregressivePolicy
:canonical: src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.eval

```{autodoc2-docstring} src.models.common.non_autoregressive.policy.NonAutoregressivePolicy.eval
```

````

`````
