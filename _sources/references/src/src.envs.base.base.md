# {py:mod}`src.envs.base.base`

```{py:module} src.envs.base.base
```

```{autodoc2-docstring} src.envs.base.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RL4COEnvBase <src.envs.base.base.RL4COEnvBase>`
  - ```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase
    :summary:
    ```
````

### API

`````{py:class} RL4COEnvBase(generator: typing.Optional[typing.Any] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', batch_size: typing.Optional[typing.Union[torch.Size, int]] = None, **kwargs: typing.Any)
:canonical: src.envs.base.base.RL4COEnvBase

Bases: {py:obj}`src.envs.base.batch.BatchMixin`, {py:obj}`src.envs.base.ops.OpsMixin`, {py:obj}`torchrl.envs.EnvBase`

```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase.__init__
```

````{py:attribute} name
:canonical: src.envs.base.base.RL4COEnvBase.name
:type: str
:value: >
   'base'

```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.base.base.RL4COEnvBase.node_dim
:type: int
:value: >
   3

```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase.node_dim
```

````

````{py:property} dim
:canonical: src.envs.base.base.RL4COEnvBase.dim
:type: int

```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase.dim
```

````

````{py:method} render(td: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: src.envs.base.base.RL4COEnvBase.render
:abstractmethod:

```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase.render
```

````

````{py:method} __repr__() -> str
:canonical: src.envs.base.base.RL4COEnvBase.__repr__

```{autodoc2-docstring} src.envs.base.base.RL4COEnvBase.__repr__
```

````

`````
