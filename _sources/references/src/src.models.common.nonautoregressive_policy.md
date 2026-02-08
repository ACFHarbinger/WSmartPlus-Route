# {py:mod}`src.models.common.nonautoregressive_policy`

```{py:module} src.models.common.nonautoregressive_policy
```

```{autodoc2-docstring} src.models.common.nonautoregressive_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NonAutoregressivePolicy <src.models.common.nonautoregressive_policy.NonAutoregressivePolicy>`
  - ```{autodoc2-docstring} src.models.common.nonautoregressive_policy.NonAutoregressivePolicy
    :summary:
    ```
````

### API

`````{py:class} NonAutoregressivePolicy(encoder: typing.Optional[src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder] = None, decoder: typing.Optional[src.models.common.nonautoregressive_decoder.NonAutoregressiveDecoder] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, **kwargs)
:canonical: src.models.common.nonautoregressive_policy.NonAutoregressivePolicy

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.nonautoregressive_policy.NonAutoregressivePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, strategy: str = 'sampling', num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.forward

```{autodoc2-docstring} src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.forward
```

````

````{py:method} set_strategy(strategy: str, **kwargs)
:canonical: src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.set_strategy

```{autodoc2-docstring} src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.set_strategy
```

````

````{py:method} common_decoding(strategy: str, td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, heatmap: torch.Tensor, actions: typing.Optional[torch.Tensor] = None, **decoding_kwargs)
:canonical: src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.common_decoding

```{autodoc2-docstring} src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.common_decoding
```

````

````{py:method} eval()
:canonical: src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.eval

```{autodoc2-docstring} src.models.common.nonautoregressive_policy.NonAutoregressivePolicy.eval
```

````

`````
