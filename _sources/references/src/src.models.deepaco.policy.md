# {py:mod}`src.models.deepaco.policy`

```{py:module} src.models.deepaco.policy
```

```{autodoc2-docstring} src.models.deepaco.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepACOPolicy <src.models.deepaco.policy.DeepACOPolicy>`
  - ```{autodoc2-docstring} src.models.deepaco.policy.DeepACOPolicy
    :summary:
    ```
````

### API

`````{py:class} DeepACOPolicy(encoder: typing.Optional[logic.src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder] = None, decoder: typing.Optional[logic.src.models.subnets.decoders.deepaco.ACODecoder] = None, embed_dim: int = 128, num_encoder_layers: int = 3, num_heads: int = 8, n_ants: int = 20, n_iterations: int = 1, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, use_local_search: bool = True, env_name: typing.Optional[str] = None, **kwargs)
:canonical: src.models.deepaco.policy.DeepACOPolicy

Bases: {py:obj}`logic.src.models.common.nonautoregressive_policy.NonAutoregressivePolicy`

```{autodoc2-docstring} src.models.deepaco.policy.DeepACOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.deepaco.policy.DeepACOPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.deepaco.policy.DeepACOPolicy.forward

```{autodoc2-docstring} src.models.deepaco.policy.DeepACOPolicy.forward
```

````

`````
