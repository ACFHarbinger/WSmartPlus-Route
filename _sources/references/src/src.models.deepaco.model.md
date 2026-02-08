# {py:mod}`src.models.deepaco.model`

```{py:module} src.models.deepaco.model
```

```{autodoc2-docstring} src.models.deepaco.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepACO <src.models.deepaco.model.DeepACO>`
  - ```{autodoc2-docstring} src.models.deepaco.model.DeepACO
    :summary:
    ```
````

### API

`````{py:class} DeepACO(embed_dim: int = 128, num_encoder_layers: int = 3, num_heads: int = 8, n_ants: int = 20, n_iterations: int = 1, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, use_local_search: bool = True, baseline: str = 'rollout', env_name: typing.Optional[str] = None, **kwargs)
:canonical: src.models.deepaco.model.DeepACO

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.deepaco.model.DeepACO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.deepaco.model.DeepACO.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.deepaco.model.DeepACO.forward

```{autodoc2-docstring} src.models.deepaco.model.DeepACO.forward
```

````

````{py:method} set_decode_type(decode_type: str, **kwargs)
:canonical: src.models.deepaco.model.DeepACO.set_decode_type

```{autodoc2-docstring} src.models.deepaco.model.DeepACO.set_decode_type
```

````

````{py:method} eval()
:canonical: src.models.deepaco.model.DeepACO.eval

```{autodoc2-docstring} src.models.deepaco.model.DeepACO.eval
```

````

`````
