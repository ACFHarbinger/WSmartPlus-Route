# {py:mod}`src.models.policies.selection.combined`

```{py:module} src.models.policies.selection.combined
```

```{autodoc2-docstring} src.models.policies.selection.combined
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CombinedSelector <src.models.policies.selection.combined.CombinedSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.combined.CombinedSelector
    :summary:
    ```
````

### API

`````{py:class} CombinedSelector(selectors: list[src.models.policies.selection.base.VectorizedSelector], logic: str = 'or')
:canonical: src.models.policies.selection.combined.CombinedSelector

Bases: {py:obj}`src.models.policies.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.models.policies.selection.combined.CombinedSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.selection.combined.CombinedSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.combined.CombinedSelector.select

```{autodoc2-docstring} src.models.policies.selection.combined.CombinedSelector.select
```

````

`````
