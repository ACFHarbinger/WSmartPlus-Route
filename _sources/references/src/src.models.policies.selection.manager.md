# {py:mod}`src.models.policies.selection.manager`

```{py:module} src.models.policies.selection.manager
```

```{autodoc2-docstring} src.models.policies.selection.manager
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ManagerSelector <src.models.policies.selection.manager.ManagerSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.manager.ManagerSelector
    :summary:
    ```
````

### API

`````{py:class} ManagerSelector(manager=None, manager_config: typing.Optional[dict] = None, threshold: float = 0.5, device: str = 'cuda')
:canonical: src.models.policies.selection.manager.ManagerSelector

Bases: {py:obj}`src.models.policies.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.models.policies.selection.manager.ManagerSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.selection.manager.ManagerSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, locs: typing.Optional[torch.Tensor] = None, waste_history: typing.Optional[torch.Tensor] = None, threshold: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.manager.ManagerSelector.select

```{autodoc2-docstring} src.models.policies.selection.manager.ManagerSelector.select
```

````

````{py:method} load_weights(path: str)
:canonical: src.models.policies.selection.manager.ManagerSelector.load_weights

```{autodoc2-docstring} src.models.policies.selection.manager.ManagerSelector.load_weights
```

````

`````
