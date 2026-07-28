# {py:mod}`src.policies.vector.selection.combined`

```{py:module} src.policies.vector.selection.combined
```

```{autodoc2-docstring} src.policies.vector.selection.combined
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CombinedSelector <src.policies.vector.selection.combined.CombinedSelector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.combined.CombinedSelector
    :summary:
    ```
````

### API

`````{py:class} CombinedSelector(selectors: typing.List[src.policies.vector.selection.base.VectorizedSelector], logic: str = 'or')
:canonical: src.policies.vector.selection.combined.CombinedSelector

Bases: {py:obj}`src.policies.vector.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.vector.selection.combined.CombinedSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.vector.selection.combined.CombinedSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.vector.selection.combined.CombinedSelector.select

```{autodoc2-docstring} src.policies.vector.selection.combined.CombinedSelector.select
```

````

`````
