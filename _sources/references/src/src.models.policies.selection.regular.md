# {py:mod}`src.models.policies.selection.regular`

```{py:module} src.models.policies.selection.regular
```

```{autodoc2-docstring} src.models.policies.selection.regular
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RegularSelector <src.models.policies.selection.regular.RegularSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.regular.RegularSelector
    :summary:
    ```
````

### API

`````{py:class} RegularSelector(frequency: int = 3)
:canonical: src.models.policies.selection.regular.RegularSelector

Bases: {py:obj}`src.models.policies.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.models.policies.selection.regular.RegularSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.selection.regular.RegularSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, current_day: typing.Optional[torch.Tensor] = None, frequency: typing.Optional[int] = None, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.regular.RegularSelector.select

```{autodoc2-docstring} src.models.policies.selection.regular.RegularSelector.select
```

````

`````
