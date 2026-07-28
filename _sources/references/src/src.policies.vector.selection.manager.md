# {py:mod}`src.policies.vector.selection.manager`

```{py:module} src.policies.vector.selection.manager
```

```{autodoc2-docstring} src.policies.vector.selection.manager
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ManagerSelector <src.policies.vector.selection.manager.ManagerSelector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.manager.ManagerSelector
    :summary:
    ```
````

### API

`````{py:class} ManagerSelector(manager: typing.Optional[logic.src.models.meta.hrl_manager.MandatoryManager] = None, manager_config: typing.Optional[typing.Dict[str, typing.Any]] = None, threshold: float = 0.5, device: str = 'cuda')
:canonical: src.policies.vector.selection.manager.ManagerSelector

Bases: {py:obj}`src.policies.vector.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.vector.selection.manager.ManagerSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.vector.selection.manager.ManagerSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, locs: typing.Optional[torch.Tensor] = None, waste_history: typing.Optional[torch.Tensor] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.vector.selection.manager.ManagerSelector.select

```{autodoc2-docstring} src.policies.vector.selection.manager.ManagerSelector.select
```

````

````{py:method} load_weights(path: str) -> None
:canonical: src.policies.vector.selection.manager.ManagerSelector.load_weights

```{autodoc2-docstring} src.policies.vector.selection.manager.ManagerSelector.load_weights
```

````

`````
