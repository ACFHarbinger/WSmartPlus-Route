# {py:mod}`src.models.nargnn.model`

```{py:module} src.models.nargnn.model
```

```{autodoc2-docstring} src.models.nargnn.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NARGNN <src.models.nargnn.model.NARGNN>`
  - ```{autodoc2-docstring} src.models.nargnn.model.NARGNN
    :summary:
    ```
````

### API

`````{py:class} NARGNN(embed_dim: int = 64, env_name: str = 'tsp', num_layers_heatmap_generator: int = 5, num_layers_graph_encoder: int = 15, baseline: str = 'rollout', **kwargs)
:canonical: src.models.nargnn.model.NARGNN

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.nargnn.model.NARGNN
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.nargnn.model.NARGNN.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.nargnn.model.NARGNN.forward

```{autodoc2-docstring} src.models.nargnn.model.NARGNN.forward
```

````

````{py:method} set_decode_type(decode_type: str, **kwargs)
:canonical: src.models.nargnn.model.NARGNN.set_decode_type

```{autodoc2-docstring} src.models.nargnn.model.NARGNN.set_decode_type
```

````

`````
