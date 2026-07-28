# {py:mod}`src.models.core.gfacs.policy`

```{py:module} src.models.core.gfacs.policy
```

```{autodoc2-docstring} src.models.core.gfacs.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GFACSPolicy <src.models.core.gfacs.policy.GFACSPolicy>`
  - ```{autodoc2-docstring} src.models.core.gfacs.policy.GFACSPolicy
    :summary:
    ```
````

### API

`````{py:class} GFACSPolicy(encoder: typing.Optional[logic.src.models.subnets.encoders.gfacs.encoder.GFACSEncoder] = None, decoder: typing.Optional[logic.src.models.subnets.decoders.deepaco.ACODecoder] = None, embed_dim: int = 128, num_encoder_layers: int = 3, num_heads: int = 8, n_ants: int = 20, n_iterations: int = 1, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, use_local_search: bool = True, env_name: typing.Optional[str] = None, **kwargs: typing.Any)
:canonical: src.models.core.gfacs.policy.GFACSPolicy

Bases: {py:obj}`logic.src.models.core.deepaco.policy.DeepACOPolicy`

```{autodoc2-docstring} src.models.core.gfacs.policy.GFACSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.gfacs.policy.GFACSPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.base.RL4COEnvBase, num_starts: int = 1, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.gfacs.policy.GFACSPolicy.forward

```{autodoc2-docstring} src.models.core.gfacs.policy.GFACSPolicy.forward
```

````

`````
