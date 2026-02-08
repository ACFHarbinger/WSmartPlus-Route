# {py:mod}`src.models.policies.selection.last_minute`

```{py:module} src.models.policies.selection.last_minute
```

```{autodoc2-docstring} src.models.policies.selection.last_minute
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LastMinuteSelector <src.models.policies.selection.last_minute.LastMinuteSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.last_minute.LastMinuteSelector
    :summary:
    ```
````

### API

`````{py:class} LastMinuteSelector(threshold: float = 0.7)
:canonical: src.models.policies.selection.last_minute.LastMinuteSelector

Bases: {py:obj}`src.models.policies.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.models.policies.selection.last_minute.LastMinuteSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.selection.last_minute.LastMinuteSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, threshold: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.last_minute.LastMinuteSelector.select

```{autodoc2-docstring} src.models.policies.selection.last_minute.LastMinuteSelector.select
```

````

`````
