# {py:mod}`src.policies.mandatory_selection.vectorized.combined`

```{py:module} src.policies.mandatory_selection.vectorized.combined
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.combined
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CombinedSelector <src.policies.mandatory_selection.vectorized.combined.CombinedSelector>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.combined.CombinedSelector
    :summary:
    ```
````

### API

`````{py:class} CombinedSelector(selectors: typing.List[src.policies.mandatory_selection.vectorized.base.VectorizedSelector], logic: str = 'or')
:canonical: src.policies.mandatory_selection.vectorized.combined.CombinedSelector

Bases: {py:obj}`src.policies.mandatory_selection.vectorized.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.combined.CombinedSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.combined.CombinedSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.mandatory_selection.vectorized.combined.CombinedSelector.select

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.combined.CombinedSelector.select
```

````

`````
