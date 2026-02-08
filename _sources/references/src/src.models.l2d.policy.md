# {py:mod}`src.models.l2d.policy`

```{py:module} src.models.l2d.policy
```

```{autodoc2-docstring} src.models.l2d.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`L2DPolicy <src.models.l2d.policy.L2DPolicy>`
  - ```{autodoc2-docstring} src.models.l2d.policy.L2DPolicy
    :summary:
    ```
````

### API

`````{py:class} L2DPolicy(embed_dim: int = 128, num_encoder_layers: int = 3, feedforward_hidden: int = 512, env_name: str = 'jssp', temp: float = 1.0, tanh_clipping: float = 10.0, train_decode_type: str = 'sampling', val_decode_type: str = 'greedy', test_decode_type: str = 'greedy', **kwargs)
:canonical: src.models.l2d.policy.L2DPolicy

Bases: {py:obj}`logic.src.models.common.constructive.ConstructivePolicy`

```{autodoc2-docstring} src.models.l2d.policy.L2DPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.l2d.policy.L2DPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[typing.Any] = None, decode_type: str = 'sampling', num_starts: int = 1, **kwargs) -> dict
:canonical: src.models.l2d.policy.L2DPolicy.forward

```{autodoc2-docstring} src.models.l2d.policy.L2DPolicy.forward
```

````

`````
