# {py:mod}`src.policies.mandatory_selection.vectorized.regular`

```{py:module} src.policies.mandatory_selection.vectorized.regular
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.regular
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RegularSelector <src.policies.mandatory_selection.vectorized.regular.RegularSelector>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.regular.RegularSelector
    :summary:
    ```
````

### API

`````{py:class} RegularSelector(frequency: int = 3)
:canonical: src.policies.mandatory_selection.vectorized.regular.RegularSelector

Bases: {py:obj}`src.policies.mandatory_selection.vectorized.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.regular.RegularSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.regular.RegularSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, current_day: typing.Optional[typing.Union[torch.Tensor, int]] = None, frequency: typing.Optional[int] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.mandatory_selection.vectorized.regular.RegularSelector.select

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.regular.RegularSelector.select
```

````

`````
