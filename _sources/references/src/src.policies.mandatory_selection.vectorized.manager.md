# {py:mod}`src.policies.mandatory_selection.vectorized.manager`

```{py:module} src.policies.mandatory_selection.vectorized.manager
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.manager
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ManagerSelector <src.policies.mandatory_selection.vectorized.manager.ManagerSelector>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.manager.ManagerSelector
    :summary:
    ```
````

### API

`````{py:class} ManagerSelector(manager: typing.Optional[logic.src.models.meta.hrl_manager.MandatoryManager] = None, manager_config: typing.Optional[typing.Dict[str, typing.Any]] = None, threshold: float = 0.5, device: str = 'cuda')
:canonical: src.policies.mandatory_selection.vectorized.manager.ManagerSelector

Bases: {py:obj}`src.policies.mandatory_selection.vectorized.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.manager.ManagerSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.manager.ManagerSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, locs: typing.Optional[torch.Tensor] = None, waste_history: typing.Optional[torch.Tensor] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.mandatory_selection.vectorized.manager.ManagerSelector.select

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.manager.ManagerSelector.select
```

````

````{py:method} load_weights(path: str) -> None
:canonical: src.policies.mandatory_selection.vectorized.manager.ManagerSelector.load_weights

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.manager.ManagerSelector.load_weights
```

````

`````
