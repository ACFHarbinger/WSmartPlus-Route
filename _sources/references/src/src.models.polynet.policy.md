# {py:mod}`src.models.polynet.policy`

```{py:module} src.models.polynet.policy
```

```{autodoc2-docstring} src.models.polynet.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolyNetPolicy <src.models.polynet.policy.PolyNetPolicy>`
  - ```{autodoc2-docstring} src.models.polynet.policy.PolyNetPolicy
    :summary:
    ```
````

### API

`````{py:class} PolyNetPolicy(k: int, encoder: typing.Optional[torch.nn.Module] = None, encoder_type: str = 'AM', embed_dim: int = 128, num_encoder_layers: int = 6, num_heads: int = 8, normalization: str = 'instance', feedforward_hidden: int = 512, env_name: str = 'vrpp', temperature: float = 1.0, tanh_clipping: float = 10.0, mask_logits: bool = True, train_strategy: str = 'sampling', val_strategy: str = 'sampling', test_strategy: str = 'sampling', **kwargs)
:canonical: src.models.polynet.policy.PolyNetPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.polynet.policy.PolyNetPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.polynet.policy.PolyNetPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, phase: str = 'train', return_actions: bool = True, num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.polynet.policy.PolyNetPolicy.forward

```{autodoc2-docstring} src.models.polynet.policy.PolyNetPolicy.forward
```

````

`````
