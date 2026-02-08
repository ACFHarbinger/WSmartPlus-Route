# {py:mod}`src.models.common.transductive_base`

```{py:module} src.models.common.transductive_base
```

```{autodoc2-docstring} src.models.common.transductive_base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TransductiveModel <src.models.common.transductive_base.TransductiveModel>`
  - ```{autodoc2-docstring} src.models.common.transductive_base.TransductiveModel
    :summary:
    ```
````

### API

`````{py:class} TransductiveModel(model: torch.nn.Module, optimizer_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, n_search_steps: int = 10, **kwargs: typing.Any)
:canonical: src.models.common.transductive_base.TransductiveModel

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.transductive_base.TransductiveModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.transductive_base.TransductiveModel.__init__
```

````{py:method} _setup_optimizer(params: typing.Any) -> torch.optim.Optimizer
:canonical: src.models.common.transductive_base.TransductiveModel._setup_optimizer

```{autodoc2-docstring} src.models.common.transductive_base.TransductiveModel._setup_optimizer
```

````

````{py:method} _get_search_params() -> typing.Any
:canonical: src.models.common.transductive_base.TransductiveModel._get_search_params

```{autodoc2-docstring} src.models.common.transductive_base.TransductiveModel._get_search_params
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.common.transductive_base.TransductiveModel.forward

```{autodoc2-docstring} src.models.common.transductive_base.TransductiveModel.forward
```

````

`````
